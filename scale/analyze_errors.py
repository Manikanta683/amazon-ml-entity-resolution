#!/usr/bin/env python3
"""Tune-set error analysis: where does the winner lose F_0.5?"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ART, TMP, log

with open(os.path.join(ART, "model.pkl"), "rb") as f:
    model = pickle.load(f)
thr = float(open(os.path.join(ART, "threshold.txt")).read().strip())

for shard in ("india", "us"):
    d = np.load(os.path.join(TMP, f"feat_tune_{shard}.npz"), allow_pickle=True)
    X, y, ent = d["X"], d["y"], d["ent"]
    gt_counts = d["gt_counts"]
    n_ent = len(gt_counts)
    prob = model.predict_proba(X)[:, 1]
    pred = prob >= thr

    # per-entity outcome classes
    tp_e = np.bincount(ent[pred & (y == 1)], minlength=n_ent)
    fp_e = np.bincount(ent[pred & (y == 0)], minlength=n_ent)
    fn_e = gt_counts - tp_e
    has_gt = gt_counts > 0
    # perfect entities (F=1)
    perfect = ((tp_e == gt_counts) & (fp_e == 0))
    # entities with missed dups (recall loss)
    recall_loss = has_gt & (fn_e > 0)
    # entities with FPs (precision loss)
    prec_loss = fp_e > 0
    # singleton FPs
    sing_fp = (~has_gt) & (fp_e > 0)
    log(f"[{shard}] entities={n_ent:,} perfect={perfect.mean():.3f} "
        f"recall_loss={recall_loss.mean():.3f} prec_loss={prec_loss.mean():.3f} "
        f"singleton_fp={sing_fp.sum():,}")
    # score distribution of FPs vs TPs
    fp_scores = prob[pred & (y == 0)]
    tp_scores = prob[pred & (y == 1)]
    fn_scores = prob[(~pred) & (y == 1)]
    log(f"[{shard}] FP scores: med={np.median(fp_scores):.3f} "
        f"p90={np.percentile(fp_scores,90):.3f} n={len(fp_scores):,}")
    log(f"[{shard}] TP scores: med={np.median(tp_scores):.3f} "
        f"p10={np.percentile(tp_scores,10):.3f} n={len(tp_scores):,}")
    log(f"[{shard}] FN scores (missed dups): med={np.median(fn_scores):.3f} "
        f"n={len(fn_scores):,} ({len(fn_scores)/max(1,(y==1).sum()):.3f} of positives)")
