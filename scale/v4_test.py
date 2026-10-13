#!/usr/bin/env python3
"""v4 full test scoring.

Two-pass inference, NO confidence gate (v3's gate wrongly assumed
confident => complete and skipped stage-2 for 74% of entities that needed
it):
  Pass 1: TF-IDF stage-1 candidates (positions 0-14) for ALL S1
  Pass 2: exact-leg candidates (positions 15+, transliteration-folded +
          v3_norm exact-name legs) for ALL S1
Both passes use the same 33-feature HGB (trained on the hard-positive
distribution); global threshold decision per pair.

Reads tmp/test_{shard}_cand64v4.npz (built by v4_block_aug --augment-test).
Writes output_v4/matching_results.tsv (test file order) and
output_v4/candidate_pairs.tsv (the exact inference set per S1 — kept tight,
~15-25/S1, for the smaller-set tiebreaker).

Memory: 10k-S1 chunks; per-chunk pool field loading; mmap'd npz input.
"""
import os
import sys
import gc
import glob
import pickle
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, read_chunks, log  # noqa: E402
from v4_featurize import featurize_chunk_v4  # noqa: E402

V4_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v4")
OUT_V4 = os.path.expanduser("~/workspace/amazon_ml/output_v4")
CHUNK = 10000
STAGE1_W = 15


def load_model():
    m = pickle.load(open(os.path.join(V4_ART, "model.pkl"), "rb"))
    thr = float(open(os.path.join(V4_ART, "threshold.txt")).read().strip())
    fd = pickle.load(open(os.path.join(V4_ART, "fold_dict.pkl"), "rb"))
    return m["hgb"], thr, fd["tmap"], set(fd["dev_suffix"])


def cos_name_addr(vn, va, s1_names, s1_addrs, pool_names, pool_addrs, pairs):
    An = vn.transform(s1_names)
    Bn = vn.transform(pool_names).tocsr()
    Aa = va.transform(s1_addrs)
    Ba = va.transform(pool_addrs).tocsr()
    cn = np.empty(len(pairs), dtype=np.float32)
    ca = np.empty(len(pairs), dtype=np.float32)
    by_pos = {}
    for k, (p, r) in enumerate(pairs):
        by_pos.setdefault(p, []).append((k, r))
    for p, krs in by_pos.items():
        ks = [k for k, _ in krs]
        rs = [r for _, r in krs]
        sn = np.asarray(An.getrow(p).dot(Bn[rs].T).todense()).ravel()
        sa = np.asarray(Aa.getrow(p).dot(Ba[rs].T).todense()).ravel()
        for k, x, y in zip(ks, sn, sa):
            cn[k] = x
            ca[k] = y
    return cn, ca


def score_pass(s1_recs, rows_list, cos_list, pn, pa, pc, vn, va, hgb,
               tmap, dev_suffix):
    """33-feature HGB scoring for one pass. Returns (prob, pairs)."""
    pairs = []
    for pos, rows in enumerate(rows_list):
        for r in rows:
            pairs.append((pos, r))
    if not pairs:
        return np.zeros(0, dtype=np.float32), []
    cn, ca = cos_name_addr(vn, va,
                           [r[0] for r in s1_recs], [r[1] for r in s1_recs],
                           pn, pa, pairs)
    cn_l, ca_l, k = [], [], 0
    for rows in rows_list:
        cn_l.append(cn[k:k + len(rows)])
        ca_l.append(ca[k:k + len(rows)])
        k += len(rows)
    Xb, pair_s1 = featurize_chunk_v4(s1_recs, rows_list, cos_list, pn, pa, pc,
                                     cn_l, ca_l, n_jobs=2, fold_map=tmap,
                                     dev_suffix=dev_suffix)
    prob = hgb.predict_proba(Xb)[:, 1].astype(np.float32)
    del Xb
    gc.collect()
    return prob, pairs


def load_pool_fields_chunk(pool_ids, ref_rows):
    need = set(pool_ids[r] for r in ref_rows)
    pf = {}
    for src in ("test_source2", "test_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
            for eid, nm, ad, cy in zip(chunk["entity_id"], chunk["name_norm"],
                                       chunk["address_norm"],
                                       chunk["country_norm"]):
                if eid in need:
                    pf[eid] = (nm, ad, cy)
                    if len(pf) >= len(need):
                        break
            if len(pf) >= len(need):
                break
        if len(pf) >= len(need):
            break
    return pf


def score_shard(shard, hgb, thr, tmap, dev_suffix, out_lines, cand_lines):
    npz_path = os.path.join(TMP, f"test_{shard}_cand64v4.npz")
    d = np.load(npz_path, mmap_mode="r")
    s1_ids = [str(x) for x in np.asarray(d["s1_ids"])]
    top_rows, top_cos = d["top_rows"], d["top_cos"]
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in np.asarray(d["pool_ids"])])
    n = len(s1_ids)
    log(f"[{shard}] {n:,} S1")
    vn = pickle.load(open(os.path.join(V4_ART, f"vec_name_{shard}.pkl"), "rb"))
    va = pickle.load(open(os.path.join(V4_ART, f"vec_addr_{shard}.pkl"), "rb"))

    want = set(s1_ids)
    s1f = {}
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv")):
        m = chunk["entity_id"].isin(want)
        sub = chunk[m]
        for eid, nm, ad, cy in zip(sub["entity_id"], sub["name_norm"],
                                   sub["address_norm"], sub["country_norm"]):
            s1f[eid] = (nm, ad, cy)
        if len(s1f) >= len(want):
            break

    for c0 in range(0, n, CHUNK):
        c1 = min(n, c0 + CHUNK)
        # split stage-1 (0:15) vs legs (15:) — same model, no gate
        rows1, cos1, rows2, cos2, ref = [], [], [], [], set()
        for j in range(c0, c1):
            r = [int(x) for x in top_rows[j] if int(x) >= 0]
            c = [float(top_cos[j, t]) for t in range(len(r))]
            r1, c1_ = r[:STAGE1_W], c[:STAGE1_W]
            r2, c2_ = r[STAGE1_W:], c[STAGE1_W:]
            rows1.append(r1); cos1.append(c1_)
            rows2.append(r2); cos2.append(c2_)
            ref.update(r)
        ref = sorted(ref)
        rmap = {r: i for i, r in enumerate(ref)}
        rpids = [pool_ids[r] for r in ref]
        pf = load_pool_fields_chunk(pool_ids, ref)
        pn = [pf[pid][0] for pid in rpids]
        pa = [pf[pid][1] for pid in rpids]
        pc = [pf[pid][2] for pid in rpids]
        del pf
        gc.collect()

        s1_recs = [s1f[s1_ids[j]] for j in range(c0, c1)]
        m1 = [[rmap[r] for r in rows] for rows in rows1]
        m2 = [[rmap[r] for r in rows] for rows in rows2]
        prob1, _ = score_pass(s1_recs, m1, cos1, pn, pa, pc, vn, va, hgb,
                              tmap, dev_suffix)
        prob2, _ = score_pass(s1_recs, m2, cos2, pn, pa, pc, vn, va, hgb,
                              tmap, dev_suffix)

        k1 = k2 = 0
        for pos in range(c1 - c0):
            j = c0 + pos
            sid = s1_ids[j]
            cand = []
            for t, r in enumerate(rows1[pos]):
                cand.append((rpids[rmap[r]], float(prob1[k1 + t])))
            k1 += len(rows1[pos])
            for t, r in enumerate(rows2[pos]):
                cand.append((rpids[rmap[r]], float(prob2[k2 + t])))
            k2 += len(rows2[pos])
            chosen = [pid for pid, p in sorted(cand, key=lambda x: -x[1])
                      if p >= thr]
            out_lines.append(f"{sid}\t{','.join(chosen)}\n")
            seen = set()
            uniq = [pid for pid, _ in cand
                    if not (pid in seen or seen.add(pid))]
            cand_lines.append(f"{sid}\t{','.join(uniq)}\n")
        del pn, pa, pc, prob1, prob2
        gc.collect()
        log(f"[{shard}] done {c1:,}/{n:,}")

    d.close()
    del vn, va
    gc.collect()


def main():
    r = subprocess.run(["pgrep", "-f", "v3_block_full.py"], capture_output=True)
    if r.returncode == 0:
        log("REFUSING: v3_block_full.py is alive (OOM risk). Exiting.")
        sys.exit(2)
    hgb, thr, tmap, dev_suffix = load_model()
    log(f"model loaded (thr={thr})")
    os.makedirs(OUT_V4, exist_ok=True)
    out_lines, cand_lines = [], []
    shards = sorted(
        os.path.basename(p)[5:-13]  # test_<shard>_cand64v4.npz
        for p in glob.glob(os.path.join(TMP, "test_*_cand64v4.npz")))
    if not shards:
        log("ERROR: no tmp/test_*_cand64v4.npz found. Run v4_block_aug "
            "--augment-test first.")
        sys.exit(2)
    log(f"shards: {shards}")
    for shard in shards:
        score_shard(shard, hgb, thr, tmap, dev_suffix, out_lines,
                    cand_lines)
    order = []
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv"),
                             usecols=["entity_id"]):
        order.extend(chunk["entity_id"].tolist())
    mo, mc = {}, {}
    for line in out_lines:
        sid, _, mids = line.rstrip("\n").partition("\t")
        mo[sid] = mids
    for line in cand_lines:
        sid, _, cids = line.rstrip("\n").partition("\t")
        mc[sid] = cids
    with open(os.path.join(OUT_V4, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{mo.get(sid, '')}\n")
    with open(os.path.join(OUT_V4, "candidate_pairs.tsv"), "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{mc.get(sid, '')}\n")
    log(f"wrote {OUT_V4}/matching_results.tsv + candidate_pairs.tsv")


if __name__ == "__main__":
    main()
