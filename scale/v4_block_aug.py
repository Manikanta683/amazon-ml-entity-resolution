#!/usr/bin/env python3
"""v4 augmented blocking.

Legs (union, dedup, cap):
  L1  v3 dual-index stage-1 top-15 (word TF-IDF + exact-name extras) — from npz
  L2  transliteration-folded exact-name leg: ALL pool rows whose folded name
      (Devanagari tokens mapped via mined tmap, suffix-stripped) equals the
      S1's folded name
  L3  v3_norm exact-name leg: ALL pool rows whose v3_norm(name) equals the
      S1's v3_norm(name) (catches aggarwal/agarwal-class variants missed by
      the raw-name map)

The generic TF-IDF stage-2 (rows 16-100) is intentionally NOT carried into
the v4 test npz: it is 85 mostly-negative candidates per S1 whose recoverable
positives are dominated by transliterations (caught by L2 with higher
precision). Sample validation measures the residual before the full run.

CLI:
  --mine            run transliteration mining (v4_mine)
  --augment-test    build tmp/test_{shard}_cand64v4.npz from v3 cand100 npz
                    (+ copy name/addr vecs artifacts_v3 -> artifacts_v4)

Library (for v4_train.py):
  build_train_block(shard, s1_ids, s1_texts, s1_names, out_path)
      dual-index two-stage top-100 on the TRAIN pool + L2/L3 legs -> npz
"""
import os
import sys
import gc
import pickle
import shutil
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, read_chunks, log  # noqa: E402
from v3_featurize import v3_norm  # noqa: E402
from block import discover_shards  # noqa: E402
from v3_block import ShardIndexV3, query_two_stage  # noqa: E402

V4_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v4")
V3_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v3")

V4_FILL_TO = 64
V4_TRAIN_FILL_TO = 128
LAT_SUFFIX = {"private", "limited", "pvt", "ltd", "inc", "corp", "corporation",
              "llc", "llp", "co", "company"}


def load_fold_dict():
    with open(os.path.join(V4_ART, "fold_dict.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["tmap"], set(d["dev_suffix"])


def fold_name(name, tmap, dev_suffix=frozenset()):
    """Transliteration-fold + suffix-strip a normalized name."""
    toks = []
    for t in name.split():
        if t in dev_suffix:
            continue
        if any("\u0900" <= c <= "\u097f" for c in t):
            t = tmap.get(t, t)
        toks.append(t)
    while toks and toks[-1] in LAT_SUFFIX:
        toks.pop()
    return " ".join(toks)


def build_name_maps(pool_paths, shard, shard_map, tmap, dev_suffix,
                    want_folded, want_v3n):
    """Stream pool; index rows by folded/v3n name (only names in want_*)."""
    fold_map = defaultdict(list)
    v3n_map = defaultdict(list)
    pool_ids = []
    for path in pool_paths:
        for chunk in read_chunks(path,
                                 usecols=["entity_id", "name_norm",
                                          "country_norm"]):
            m = (chunk["country_norm"].map(
                lambda c: shard_map.get(c, "other")) == shard)
            sub = chunk[m]
            if len(sub) == 0:
                continue
            base = len(pool_ids)
            eids = sub["entity_id"].tolist()
            nms = sub["name_norm"].tolist()
            pool_ids.extend(eids)
            for j, nm in enumerate(nms):
                if not nm:
                    continue
                r = base + j
                fn = fold_name(nm, tmap, dev_suffix)
                if fn and fn in want_folded:
                    fold_map[fn].append(r)
                vn = v3_norm(nm)
                if vn and vn in want_v3n and vn != nm:
                    # vn == nm adds nothing over the raw exact-name leg
                    v3n_map[vn].append(r)
    pool_ids = np.array([e.encode() for e in pool_ids])
    log(f"[{shard}] name maps: pool={len(pool_ids):,} "
        f"fold-keys={len(fold_map):,} v3n-keys={len(v3n_map):,}")
    return pool_ids, dict(fold_map), dict(v3n_map)


def shard_s1_folded(s1_path, shard, shard_map, tmap, dev_suffix, want_ids=None):
    """{eid: (folded_name, v3n_name)} for a shard's S1 rows."""
    d = {}
    for chunk in read_chunks(s1_path,
                             usecols=["entity_id", "name_norm", "country_norm"]):
        m = (chunk["country_norm"].map(
            lambda c: shard_map.get(c, "other")) == shard)
        sub = chunk[m]
        if len(sub) == 0:
            continue
        for eid, nm in zip(sub["entity_id"], sub["name_norm"]):
            if want_ids is not None and eid not in want_ids:
                continue
            d[eid] = (fold_name(nm, tmap, dev_suffix), v3_norm(nm))
    return d


def augment_npz(in_path, out_path, pool_ids, fold_map, v3n_map, s1_leg,
                fill_to=V4_FILL_TO, chunk=100_000):
    """Add L2/L3 legs to a cand npz. s1_leg: {eid: (folded, v3n)}."""
    z = np.load(in_path, mmap_mode="r")
    s1_ids = np.array([str(x) for x in z["s1_ids"]])
    top_rows, top_cos = z["top_rows"], z["top_cos"]
    n, w = top_rows.shape
    log(f"augmenting {in_path}: n={n:,} width={w} -> {fill_to}")
    out_rows, out_cos = [], []
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        br = np.full((c1 - c0, fill_to), -1, dtype=np.int32)
        bc = np.zeros((c1 - c0, fill_to), dtype=np.float32)
        # NOTE: top_rows may be mmap int32 with -1 padding; copy to RAM chunk
        tr = np.asarray(top_rows[c0:c1])
        tc = np.asarray(top_cos[c0:c1])
        for i in range(c1 - c0):
            base_rows, base_cos = [], []
            for t in range(min(15, w)):
                r = int(tr[i, t])
                if r < 0:
                    break
                base_rows.append(r)
                base_cos.append(float(tc[i, t]))
            rset = set(base_rows)
            fn, vn = s1_leg.get(s1_ids[c0 + i], ("", ""))
            if fn:
                for r in fold_map.get(fn, ()):
                    if r not in rset:
                        rset.add(r)
                        base_rows.append(r)
                        base_cos.append(0.0)
            if vn:
                for r in v3n_map.get(vn, ()):
                    if r not in rset:
                        rset.add(r)
                        base_rows.append(r)
                        base_cos.append(0.0)
            k = min(len(base_rows), fill_to)
            br[i, :k] = base_rows[:k]
            bc[i, :k] = base_cos[:k]
        out_rows.append(br)
        out_cos.append(bc)
        del br, bc, tr, tc
        gc.collect()
        log(f"  augmented {c1:,}/{n:,}")
    top_rows_n = np.vstack(out_rows)
    top_cos_n = np.vstack(out_cos)
    del out_rows, out_cos
    gc.collect()
    # pool_ids for the v4 npz: reuse the ORIGINAL npz's pool_ids (row-aligned
    # with the v3 index). Leg rows reference the SAME pool row space because
    # build_name_maps streams pool in the same order as ShardIndex.
    orig_pool_ids = np.array(z["pool_ids"])
    z.close()
    assert len(orig_pool_ids) == len(pool_ids), \
        f"pool row mismatch: {len(orig_pool_ids)} vs {len(pool_ids)}"
    np.savez(out_path, s1_ids=np.array(s1_ids), top_rows=top_rows_n,
             top_cos=top_cos_n, pool_ids=orig_pool_ids)
    mean_c = (top_rows_n >= 0).sum(axis=1).mean()
    log(f"wrote {out_path} rows={top_rows_n.shape} mean_cands={mean_c:.1f}")
    del top_rows_n, top_cos_n, orig_pool_ids
    gc.collect()


def augment_test():
    """Build v4 test cand npz per shard from v3 cand100 npz."""
    import glob
    tmap, dev_suffix = load_fold_dict()
    pool_paths = [os.path.join(NORM, "test_source2.tsv"),
                  os.path.join(NORM, "test_source3.tsv")]
    s1_path = os.path.join(NORM, "test_source1.tsv")
    shard_map, _ = discover_shards(pool_paths)
    in_paths = sorted(glob.glob(os.path.join(TMP, "test_*_cand100.npz")))
    log(f"augmenting: {[os.path.basename(p) for p in in_paths]}")
    os.makedirs(V4_ART, exist_ok=True)
    for in_path in in_paths:
        shard = os.path.basename(in_path)[5:-12]  # test_<shard>_cand100.npz
        log(f"[{shard}] loading S1 folded names ...")
        s1_leg = shard_s1_folded(s1_path, shard, shard_map, tmap, dev_suffix)
        want_folded = {fn for fn, _ in s1_leg.values() if fn}
        want_v3n = {vn for _, vn in s1_leg.values() if vn}
        pool_ids, fold_map, v3n_map = build_name_maps(
            pool_paths, shard, shard_map, tmap, dev_suffix,
            want_folded, want_v3n)
        out_path = os.path.join(TMP, f"test_{shard}_cand64v4.npz")
        augment_npz(in_path, out_path, pool_ids, fold_map, v3n_map, s1_leg)
        del pool_ids, fold_map, v3n_map, s1_leg
        gc.collect()
    # name/addr vecs are pool-identical to v3's -> copy (no refit)
    for shard in [os.path.basename(p)[5:-12] for p in in_paths]:
        for kind in ("vec_name", "vec_addr"):
            src = os.path.join(V3_ART, f"{kind}_{shard}.pkl")
            dst = os.path.join(V4_ART, f"{kind}_{shard}.pkl")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                log(f"copied {kind}_{shard}.pkl -> artifacts_v4/")
    log("augment-test DONE")


def build_train_block(shard, s1_ids, s1_texts, s1_names, out_path):
    """Exact-legs-only train blocking (OOM-safe).

    The TF-IDF stage-1 query OOMs on india/us train pools (4-6M docs).
    Uses L2 (folded exact) + L3 (v3norm exact) only. Fast, no large matrix.
    """
    import subprocess
    r = subprocess.run(["pgrep", "-f", "[v]3_block_full[.]py"],
                       capture_output=True)
    if r.returncode == 0:
        log("REFUSING: v3_block_full.py is alive (OOM risk).")
        sys.exit(2)
    tmap, dev_suffix = load_fold_dict()
    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    shard_map, shard_counts = discover_shards(pool_paths)
    log(f"[{shard}] building exact train block (no TF-IDF) ...")
    # L2/L3 legs on the train pool
    s1_leg = {sid: (fold_name(nm, tmap, dev_suffix), v3_norm(nm))
              for sid, nm in zip(s1_ids, s1_names)}
    want_folded = {fn for fn, _ in s1_leg.values() if fn}
    want_v3n = {vn for _, vn in s1_leg.values() if vn}
    pool_ids, fold_map, v3n_map = build_name_maps(
        pool_paths, shard, shard_map, tmap, dev_suffix, want_folded, want_v3n)
    n = len(s1_ids)
    top_rows = np.full((n, V4_TRAIN_FILL_TO), -1, dtype=np.int32)
    top_cos = np.zeros((n, V4_TRAIN_FILL_TO), dtype=np.float32)
    for i in range(n):
        rset = set()
        rr = []
        fn, vn = s1_leg[s1_ids[i]]
        if fn:
            for r in fold_map.get(fn, ()):
                if r not in rset:
                    rset.add(r)
                    rr.append(r)
        if vn:
            for r in v3n_map.get(vn, ()):
                if r not in rset:
                    rset.add(r)
                    rr.append(r)
        k = min(len(rr), V4_TRAIN_FILL_TO)
        if k:
            top_rows[i, :k] = rr[:k]
            top_cos[i, :k] = 1.0
    np.savez(out_path, s1_ids=np.array(s1_ids), top_rows=top_rows,
             top_cos=top_cos, pool_ids=pool_ids.astype("S16"))
    mean_c = (top_rows >= 0).sum(axis=1).mean()
    log(f"[{shard}] wrote {out_path} mean cands={mean_c:.1f}")
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--augment-test", action="store_true")
    args = ap.parse_args()
    if args.mine:
        from v4_mine import main as mine_main
        mine_main()
    if args.augment_test:
        augment_test()
    if not (args.mine or args.augment_test):
        ap.print_help()


if __name__ == "__main__":
    main()
