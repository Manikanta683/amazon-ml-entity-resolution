#!/usr/bin/env python3
"""Oracle analysis: blocking ceiling vs classifier loss on the tune split.

Computes macro F_0.5 for a perfect classifier restricted to the top-15
candidates (the blocking ceiling), and compares to the actual 0.8805.
Writes to ~/workspace (persistent across VM restarts).
"""
import os
import numpy as np
from common import NORM, TMP, ART, DATASET, read_chunks, log

OUT = os.path.expanduser("~/workspace/amazon_ml/scale/analysis_oracle.txt")


def load_gt_for_ids(want):
    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in want:
                gt[sid] = set(m for m in str(mids).split(",") if m and m != "nan")
    return gt


def f05_per_entity(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    prec = tp / len(pred)
    rec = tp / len(truth)
    if prec + rec == 0:
        return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)


def main():
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    tune_ids = set(str(x) for x in sp["tune_ids"].tolist())
    log(f"tune S1: {len(tune_ids):,}")
    gt = load_gt_for_ids(tune_ids)
    log(f"GT loaded for {len(gt):,} tune S1")

    scores, n = 0.0, 0
    n_gt_pairs = n_hit = 0
    per_shard = {}
    for shard in ("india", "us"):
        d = np.load(os.path.join(TMP, f"tune_{shard}_cand.npz"), allow_pickle=True)
        s1_ids = [str(x) for x in d["s1_ids"].tolist()]
        top_rows, pool_ids = d["top_rows"], d["pool_ids"]
        s_scores, s_n, s_hit, s_tot = 0.0, 0, 0, 0
        for j, sid in enumerate(s1_ids):
            rows = [int(r) for r in top_rows[j] if r >= 0]
            cands = set(pool_ids[r].decode() for r in rows)
            truth = gt.get(sid, set())
            oracle_pred = truth & cands  # perfect classifier on candidates
            s_scores += f05_per_entity(oracle_pred, truth)
            s_n += 1
            s_hit += len(oracle_pred)
            s_tot += len(truth)
        per_shard[shard] = (s_scores / s_n, s_hit, s_tot, s_n)
        scores += s_scores
        n += s_n
        n_hit += s_hit
        n_gt_pairs += s_tot
        log(f"[{shard}] oracle F_0.5={s_scores/s_n:.4f} pair_recall={s_hit/max(s_tot,1):.4f}")

    overall = scores / n
    with open(OUT, "w") as f:
        f.write(f"oracle_blocking_ceiling_F05={overall:.4f}\n")
        f.write(f"actual_tune_F05=0.8805\n")
        f.write(f"classifier_loss={overall-0.8805:.4f}\n")
        f.write(f"pair_recall_top15={n_hit}/{n_gt_pairs}={n_hit/max(n_gt_pairs,1):.4f}\n")
        for shard, (sf, h, t, sn) in per_shard.items():
            f.write(f"{shard}: oracle_F05={sf:.4f} recall={h}/{t}={h/max(t,1):.4f} n={sn}\n")
    log(f"wrote {OUT}: oracle={overall:.4f} classifier_loss={overall-0.8805:.4f}")


if __name__ == "__main__":
    main()
