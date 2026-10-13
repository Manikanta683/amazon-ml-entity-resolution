"""Score GPU candidates with trained v4 HGB model.

Usage:
  python3 score_gpu.py --npz test_gpu_cand100.npz --art artifacts_v4 --norm ./norm

Outputs: matching_results.tsv, candidate_pairs.tsv
"""
import argparse, os, sys, pickle, gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log
from v4_featurize import featurize_chunk_v4
from v4_train import cos_for_pairs, fit_name_addr_vecs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--art", required=True, help="artifacts_v4 dir")
    ap.add_argument("--norm", required=True, help="norm TSV dir")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    log("loading model...")
    hgb = pickle.load(open(os.path.join(args.art, "model.pkl"), "rb"))["hgb"]
    thr = float(open(os.path.join(args.art, "threshold.txt")).read().strip())
    log(f"threshold={thr:.4f}")

    log("loading npz...")
    d = np.load(args.npz, allow_pickle=True)
    s1_ids = [str(x) for x in d["s1_ids"].tolist()]
    top_rows = d["top_rows"]
    top_cos = d["top_cos"]
    pool_ids = np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in d["pool_ids"].tolist()])
    n = len(s1_ids)
    log(f"S1={n:,}")

    # Load S1 fields + country
    log("loading S1 fields...")
    s1_path = os.path.join(args.norm, "test_source1.tsv")
    s1_df = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "name_norm", "address_norm"],
                        dtype=str).fillna("")
    s1_map = {str(r.entity_id): (str(r.name_norm), str(r.address_norm))
              for r in s1_df.itertuples()}
    del s1_df; gc.collect()

    # Load pool fields (S2+S3)
    log("loading pool fields...")
    pool_frames = []
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        p = os.path.join(args.norm, fn)
        for chunk in pd.read_csv(p, sep="\t", usecols=["entity_id", "name_norm", "address_norm"],
                                 chunksize=300_000, dtype=str):
            chunk = chunk.fillna("")
            pool_frames.append(chunk)
    pool_df = pd.concat(pool_frames, ignore_index=True)
    pool_map = {str(r.entity_id): (str(r.name_norm), str(r.address_norm))
                for r in pool_df.itertuples()}
    del pool_frames, pool_df; gc.collect()
    log(f"pool_map={len(pool_map):,}")

    # Fit global TF-IDF vectorizers on pool (for cosine features)
    # Use a sample to fit (faster), then transform
    log("fitting TF-IDF (sample)...")
    sample_ids = np.random.choice(pool_ids, size=min(500_000, len(pool_ids)), replace=False)
    pn_sample = [pool_map.get(pid, ("", ""))[0] for pid in sample_ids]
    pa_sample = [pool_map.get(pid, ("", ""))[1] for pid in sample_ids]
    vn, va = fit_name_addr_vecs(pn_sample, pa_sample)
    del pn_sample, pa_sample; gc.collect()

    log("scoring...")
    out_lines = []
    cand_lines = []
    batch = 2000
    for b0 in range(0, n, batch):
        b1 = min(n, b0 + batch)
        # Build recs, rl, cl, pn, pa, pc for this batch
        recs = []  # [(name, addr), ...] for S1s
        rl = []    # [[pool_row_idx, ...], ...]
        cl = []    # [[cos, ...], ...]
        # Collect unique pool rows in batch
        batch_pool_rows = set()
        for j in range(b0, b1):
            rows = [int(r) for r in top_rows[j] if int(r) >= 0]
            rl.append(rows)
            cl.append([float(top_cos[j, t]) for t in range(len(rows))])
            recs.append(s1_map.get(s1_ids[j], ("", "")))
            batch_pool_rows.update(rows)
        ref_rows = sorted(batch_pool_rows)
        rmap = {r: i for i, r in enumerate(ref_rows)}
        rpids = [pool_ids[r] for r in ref_rows]
        pn = [pool_map.get(pid, ("", ""))[0] for pid in rpids]
        pa = [pool_map.get(pid, ("", ""))[1] for pid in rpids]
        pc = ["" for _ in rpids]  # country not used in v4?
        # Remap rl to 0..len(ref_rows)
        rl_remap = [[rmap[r] for r in rows] for rows in rl]
        # TF-IDF cosines
        pairs = [(j, r) for j in range(len(recs)) for r in rl_remap[j]]
        cn = cos_for_pairs(vn, [r[0] for r in recs], pn, pairs)
        ca = cos_for_pairs(va, [r[1] for r in recs], pa, pairs)
        cn_l, ca_l, k = [], [], 0
        for rows in rl_remap:
            cn_l.append(cn[k:k+len(rows)])
            ca_l.append(ca[k:k+len(rows)])
            k += len(rows)
        # Featurize (tmap=None, dev_suffix="")
        Xb, _ = featurize_chunk_v4(recs, rl_remap, cl, pn, pa, pc, cn_l, ca_l,
                                   n_jobs=1, fold_map=None, dev_suffix="")
        proba = hgb.predict_proba(Xb)[:, 1]
        # Write outputs
        k = 0
        for j in range(b0, b1):
            sid = s1_ids[j]
            rows = rl[j]  # original pool indices
            matched = []
            cands = []
            for t, r in enumerate(rows):
                pid = pool_ids[r]
                cands.append(pid)
                if proba[k] >= thr:
                    matched.append(pid)
                k += 1
            out_lines.append(f"{sid}\t{','.join(matched)}\n")
            cand_lines.append(f"{sid}\t{','.join(cands)}\n")
        del Xb, proba
        gc.collect()
        if (b1 % 20000 == 0) or (b1 == n):
            log(f"scored {b1:,}/{n:,}")
    d.close()

    log("writing outputs...")
    order = s1_ids
    mo = {}
    for line in out_lines:
        sid, _, mids = line.rstrip("\n").partition("\t")
        mo[sid] = mids
    mc = {}
    for line in cand_lines:
        sid, _, cids = line.rstrip("\n").partition("\t")
        mc[sid] = cids
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{mo.get(sid, '')}\n")
    with open(os.path.join(args.out, "candidate_pairs.tsv"), "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in order:
            f.write(f"{sid}\t{mc.get(sid, '')}\n")
    log("DONE")

if __name__ == "__main__":
    main()
