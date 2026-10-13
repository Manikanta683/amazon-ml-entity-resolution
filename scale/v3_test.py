#!/usr/bin/env python3
"""v3 full test scoring (HEAVY - run only after v2_min.py finishes AND
v3_block_full.py has produced tmp/test_{shard}_cand100.npz).

Two-pass inference:
  Pass 1: top-15 (stage-1) candidates for ALL S1 -> 30-feature ensemble.
  Pass 2: S1s with pass-1 max prob < T_UNC ("uncertain") get their stage-2
          candidates (16-100) scored too; probs combined; per-entity
          expected-F_0.5 rule picks the final prediction set.

Writes output_v3/matching_results.tsv (test file order) and
output_v3/candidate_pairs.tsv (the exact inference set per S1: 15 for
certain entities, up to 100 for uncertain ones).

Memory: 3k-S1 chunks; per-chunk pool field loading; name/addr TF-IDF
cosines via per-shard prefit vectorizers + grouped sparse dots.
"""
import os
import sys
import gc
import pickle

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, read_chunks, log  # noqa: E402
from v3_featurize import featurize_chunk_v3  # noqa: E402

V3_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v3")
OUT_V3 = os.path.expanduser("~/workspace/amazon_ml/output_v3")
CHUNK = 10000
T_UNC = 0.95       # pass-1 max prob below this -> stage-2
T_2ND = 0.50       # ... or fewer than 2 candidates above this -> stage-2


def load_model():
    m = pickle.load(open(os.path.join(V3_ART, "model.pkl"), "rb"))
    thr = float(open(os.path.join(V3_ART, "threshold.txt")).read().strip())
    return m["hgb"], m.get("lr"), m["w_hgb"], thr


def cos_name_addr(vn, va, s1_names, s1_addrs, pool_names, pool_addrs, pairs):
    """Grouped sparse-dot cosines. pairs: list of (s1_pos, pool_pos)."""
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


def expected_f05_topk(probs):
    """Per-entity decision: top-k maximizing expected F_0.5 (k=0 allowed).
    probs: 1D array for ONE entity, sorted desc. Returns k."""
    p = np.asarray(probs, dtype=float)
    if len(p) == 0:
        return 0
    cum_tp = np.cumsum(p)
    kk = np.arange(1, len(p) + 1)
    total_p = p.sum()
    if total_p <= 0:
        return 0
    exp_prec = cum_tp / kk
    exp_rec = cum_tp / total_p
    denom = 0.25 * exp_prec + exp_rec
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0,
                     1.25 * exp_prec * exp_rec / np.maximum(denom, 1e-12), 0.0)
    f0 = float(np.prod(1.0 - p))
    return int(np.argmax(np.concatenate([[f0], f])))


def score_rows(s1_recs, rows_list, cos_list, pn, pa, pc, vn, va, hgb, lr, w):
    """Full 30-feature ensemble scoring for one batch. Returns prob array
    aligned with the flattened (s1_pos, row) pairs, plus pair->s1 mapping."""
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
    Xb, pair_s1 = featurize_chunk_v3(s1_recs, rows_list, cos_list, pn, pa, pc,
                                     cn_l, ca_l, n_jobs=2)
    ph = hgb.predict_proba(Xb)[:, 1]
    if lr is not None and w < 1.0:
        pl = lr.predict_proba(Xb)[:, 1]
        prob = (w * ph + (1 - w) * pl).astype(np.float32)
        del pl
    else:
        prob = ph.astype(np.float32)
    del Xb, ph
    gc.collect()
    return prob, pairs


def score_shard(shard, hgb, lr, w, thr, out_lines, cand_lines):
    d = np.load(os.path.join(TMP, f"test_{shard}_cand100.npz"), allow_pickle=True)
    s1_ids = [str(x) for x in d["s1_ids"].tolist()]
    top_rows = d["top_rows"]
    top_cos = d["top_cos"]
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in d["pool_ids"].tolist()])
    n = len(s1_ids)
    log(f"[{shard}] {n:,} S1")
    vn = pickle.load(open(os.path.join(V3_ART, f"vec_name_{shard}.pkl"), "rb"))
    va = pickle.load(open(os.path.join(V3_ART, f"vec_addr_{shard}.pkl"), "rb"))

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

    n_unc = 0
    for c0 in range(0, n, CHUNK):
        c1 = min(n, c0 + CHUNK)
        # ---- pass 1: stage-1 (first 15) for all ----
        ref = set()
        for j in range(c0, c1):
            for r in top_rows[j, :15]:
                if r >= 0:
                    ref.add(int(r))
        # stage-2 rows needed only for uncertain; collect after pass 1.
        ref = sorted(ref)
        rmap = {r: i for i, r in enumerate(ref)}
        rpids = [pool_ids[r] for r in ref]
        need = set(rpids)
        pf = {}
        for src in ("test_source2", "test_source3"):
            for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
                for eid, nm, ad, cy in zip(chunk["entity_id"], chunk["name_norm"],
                                           chunk["address_norm"], chunk["country_norm"]):
                    if eid in need:
                        pf[eid] = (nm, ad, cy)
                        if len(pf) >= len(need):
                            break
                if len(pf) >= len(need):
                    break
            if len(pf) >= len(need):
                break
        pn = [pf[pid][0] for pid in rpids]
        pa = [pf[pid][1] for pid in rpids]
        pc = [pf[pid][2] for pid in rpids]
        del pf
        gc.collect()

        s1_recs = [s1f[s1_ids[j]] for j in range(c0, c1)]
        rows15 = [[rmap[int(r)] for r in top_rows[j, :15] if r >= 0]
                   for j in range(c0, c1)]
        cos15 = [[float(top_cos[j, t]) for t in range(15) if top_rows[j, t] >= 0]
                 for j in range(c0, c1)]
        prob15, pairs15 = score_rows(s1_recs, rows15, cos15, pn, pa, pc,
                                     vn, va, hgb, lr, w)
        # per-S1 top-2 probs (pass 1) for the uncertainty gate
        k = 0
        uncertain = []
        for pos, rows in enumerate(rows15):
            j = c0 + pos
            if not rows:
                uncertain.append(j)
                continue
            ps = np.sort(prob15[k:k + len(rows)])[::-1]
            k += len(rows)
            if ps[0] < T_UNC or (len(ps) < 2 or ps[1] < T_2ND):
                uncertain.append(j)
        n_unc += len(uncertain)
        uset = set(uncertain)

        # ---- pass 2: stage-2 extras for uncertain S1s ----
        urows_all, prob2_all, rpids2 = [], np.zeros(0, dtype=np.float32), []
        if uncertain:
            ref2 = set()
            for j in uncertain:
                for r in top_rows[j, 15:]:
                    if r >= 0:
                        ref2.add(int(r))
            ref2 = sorted(ref2 - set(rmap))
            if ref2:
                rmap2 = {r: i for i, r in enumerate(ref2)}
                rpids2 = [pool_ids[r] for r in ref2]
                need2 = set(rpids2)
                pf2 = {}
                for src in ("test_source2", "test_source3"):
                    for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
                        for eid, nm, ad, cy in zip(
                                chunk["entity_id"], chunk["name_norm"],
                                chunk["address_norm"], chunk["country_norm"]):
                            if eid in need2:
                                pf2[eid] = (nm, ad, cy)
                                if len(pf2) >= len(need2):
                                    break
                        if len(pf2) >= len(need2):
                            break
                    if len(pf2) >= len(need2):
                        break
                pn2 = [pf2[pid][0] for pid in rpids2]
                pa2 = [pf2[pid][1] for pid in rpids2]
                pc2 = [pf2[pid][2] for pid in rpids2]
                del pf2
                gc.collect()
                urecs = [s1f[s1_ids[j]] for j in uncertain]
                urows_all = [[rmap2[int(r)] for r in top_rows[j, 15:] if r >= 0]
                             for j in uncertain]
                ucos = [[float(top_cos[j, t]) for t in range(15, 100)
                         if top_rows[j, t] >= 0] for j in uncertain]
                prob2_all, _ = score_rows(urecs, urows_all, ucos, pn2, pa2, pc2,
                                           vn, va, hgb, lr, w)
                del pn2, pa2, pc2
                gc.collect()
        # ---- decisions + output lines ----
        k = 0
        p1_slices = {}
        for pos, rows in enumerate(rows15):
            j = c0 + pos
            p1_slices[j] = (k, k + len(rows))
            k += len(rows)
        k2 = 0
        p2_slices = {}
        for ui, j in enumerate(uncertain):
            if ui < len(urows_all):
                p2_slices[j] = (k2, k2 + len(urows_all[ui]))
                k2 += len(urows_all[ui])
        for pos in range(c1 - c0):
            j = c0 + pos
            sid = s1_ids[j]
            a, b = p1_slices[j]
            cand_probs = [(rpids[rows15[pos][t]], float(prob15[a + t]))
                          for t in range(b - a)]
            if j in p2_slices:
                a2, b2 = p2_slices[j]
                urecs_pos = uncertain.index(j)
                for t in range(b2 - a2):
                    r = urows_all[urecs_pos][t]
                    cand_probs.append((rpids2[r], float(prob2_all[a2 + t])))
            # global-threshold decision (validated: ties/beats per-entity rules)
            chosen = [pid for pid, p in sorted(cand_probs, key=lambda x: -x[1])
                      if p >= thr]
            out_lines.append(f"{sid}\t{','.join(chosen)}\n")
            # candidate_pairs: exact inference set
            if j in uset:
                allc = [rpids[r] for r in rows15[pos]] + \
                       [rpids2[r] for r in urows_all[uncertain.index(j)]]
            else:
                allc = [rpids[r] for r in rows15[pos]]
            seen = set()
            uniq = [c for c in allc if not (c in seen or seen.add(c))]
            cand_lines.append(f"{sid}\t{','.join(uniq)}\n")
        del pn, pa, pc
        gc.collect()
        log(f"[{shard}] done {c1:,}/{n:,} (uncertain {n_unc:,})")

    del vn, va
    gc.collect()
    return n_unc


def main():
    # safety: refuse to run while v2_min.py is alive
    import subprocess
    r = subprocess.run(["pgrep", "-f", "v2_min.py"], capture_output=True)
    if r.returncode == 0:
        log("REFUSING: v2_min.py is still running (OOM risk). Exiting.")
        sys.exit(2)
    hgb, lr, w, thr = load_model()
    log(f"model loaded (w_hgb={w}, thr={thr})")
    os.makedirs(OUT_V3, exist_ok=True)
    out_lines, cand_lines = [], []
    import glob
    shards = sorted(
        os.path.basename(p)[5:-12]  # test_<shard>_cand100.npz
        for p in glob.glob(os.path.join(TMP, "test_*_cand100.npz")))
    log(f"shards with cand100: {shards}")
    for shard in shards:
        score_shard(shard, hgb, lr, w, thr, out_lines, cand_lines)
    # merge in test_source1 file order
    order = []
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv"),
                             usecols=["entity_id"]):
        order.extend(chunk["entity_id"].tolist())
    mo = {}
    mc = {}
    for line in out_lines:
        sid, _, mids = line.rstrip("\n").partition("\t")
        mo[sid] = mids
    for line in cand_lines:
        sid, _, cids = line.rstrip("\n").partition("\t")
        mc[sid] = cids
    with open(os.path.join(OUT_V3, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{mo.get(sid, '')}\n")
    with open(os.path.join(OUT_V3, "candidate_pairs.tsv"), "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{mc.get(sid, '')}\n")
    log(f"wrote {OUT_V3}/matching_results.tsv + candidate_pairs.tsv")


if __name__ == "__main__":
    main()
