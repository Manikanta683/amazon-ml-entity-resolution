#!/usr/bin/env python3
"""Per-shard tune analysis + per-shard threshold tuning."""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ART, TMP, macro_f05_from_counts, log
from train_model import tune_threshold

with open(os.path.join(ART, "model.pkl"), "rb") as f:
    model = pickle.load(f)

for shard in ("india", "us"):
    d = np.load(os.path.join(TMP, f"feat_tune_{shard}.npz"), allow_pickle=True)
    X, y, ent, gt_counts = d["X"], d["y"], d["ent"], d["gt_counts"]
    n_ent = len(gt_counts)
    prob = model.predict_proba(X)[:, 1]
    bt, bf, _, _ = tune_threshold(y, prob, ent, gt_counts, n_ent)
    # also: global threshold applied to this shard
    gthr = float(open(os.path.join(ART, "threshold.txt")).read().strip())
    pred = prob >= gthr
    tp = np.bincount(ent[pred & (y == 1)], minlength=n_ent)
    fp = np.bincount(ent[pred & (y == 0)], minlength=n_ent)
    fn = gt_counts - tp
    f_global = macro_f05_from_counts(tp, fp, fn)
    # singleton stats
    n_single = int((gt_counts == 0).sum())
    fp_single = int(((gt_counts == 0) & (fp > 0)).sum())
    log(f"[{shard}] n_ent={n_ent:,} pairs={len(X):,} "
        f"per-shard thr={bt:.3f} F05={bf:.4f} | "
        f"global thr={gthr:.3f} F05={f_global:.4f} | "
        f"singletons={n_single:,} w/ FP={fp_single:,}")
