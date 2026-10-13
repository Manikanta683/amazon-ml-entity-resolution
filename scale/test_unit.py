#!/usr/bin/env python3
"""Unit tests for block/featurize/threshold logic on tiny synthetic data."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NORM, macro_f05_from_counts, pick_threshold_midplateau
from block import ShardIndex
from featurize import featurize_chunk

tmp = "/tmp/scale_unit"
os.makedirs(tmp, exist_ok=True)

# tiny pool: 2 shards not needed; single shard 'india'
pool = pd.DataFrame({
    "entity_id": [f"S2-{i}" for i in range(6)],
    "name_norm": ["sharma general store", "sharma general store",
                  "patel electronics", "ram marketing",
                  "sharma general store", "unrelated biz"],
    "address_norm": ["12 mg rd pune", "12 mg rd pune",
                     "5 park st mumbai", "kh no 570 delhi",
                     "99 other ave nagpur", "1 main st delhi"],
    "country_norm": ["india"] * 6,
})
pool["text"] = (pool["name_norm"] + " " + pool["address_norm"]).str.strip()
pool_path = os.path.join(tmp, "pool.tsv")
pool.to_csv(pool_path, sep="\t", index=False)

idx = ShardIndex([pool_path], "india", {"india": "india"},
                 s1_names={"sharma general store", "patel electronics", "zzz"},
                 n_docs=6)
print("pool docs:", idx.P.shape[0], "vocab:", len(idx.vec.vocabulary_))

# query: exact-name S1 + fuzzy S1 + no-match S1
cids, ccos = idx.query(
    ["sharma general store 12 mg rd pune",
     "sharma generl store 12 mg rd pune",   # typo
     "zzz qqq xxx"],
    ["sharma general store", "sharma generl store", "zzz"],
    topn=50, cap=60,
)
for i, (ids, cos) in enumerate(zip(cids, ccos)):
    print(f"S1-{i}: {len(ids)} cands, top3={list(zip(ids[:3], [round(c,3) for c in cos[:3]]))}")
assert cids[0][0] == "S2-0", "exact match should rank first"
assert "S2-0" in cids[1] and "S2-1" in cids[1], "typo query should find sharma rows"
assert len(cids[2]) == 0, "no-match S1 should get zero candidates"
print("block OK")

# featurize
s1_recs = [("sharma general store", "12 mg rd pune", "india")]
X, ps = featurize_chunk(
    s1_recs, [[0, 2, 4]], [[0.99, 0.5, 0.4]],
    pool["name_norm"].tolist(), pool["address_norm"].tolist(),
    pool["country_norm"].tolist(), n_jobs=1)
print("X shape:", X.shape, "row0:", np.round(X[0], 2))
assert X.shape == (3, 16)
assert X[0, 0] == 100.0 and X[0, 11] > 90, "exact pair should score ~100"
print("featurize OK")

# threshold tuning: ent0: 1 TP avail (clean at high thr); ent1: 2 TP avail
y_true = np.array([1, 0, 1, 1])
y_prob = np.array([0.9, 0.3, 0.8, 0.7])
ent_idx = np.array([0, 0, 1, 1])
gt_counts = np.array([1, 2])
n_ent = 2
from train_model import tune_threshold
bt, bf, ths, scs = tune_threshold(y_true, y_prob, ent_idx, gt_counts, n_ent)
print(f"tune: thr={bt:.3f} F05={bf:.4f}")
# at thr in (0.3, 0.7]: ent0 {0.9} -> 1.0; ent1 {0.8,0.7} -> TP=2 -> 1.0 => 1.0
assert abs(bf - 1.0) < 1e-9, f"expected 1.0, got {bf}"
print("threshold OK")

# macro F05 edge cases
assert macro_f05_from_counts([0], [0], [0]) == 1.0      # singleton, empty pred
assert macro_f05_from_counts([0], [1], [0]) == 0.0      # singleton, false merge
assert macro_f05_from_counts([2], [1], [0]) == \
    1.25 * (2 / 3) / (0.25 * (2 / 3) + 1)  # P=2/3, R=1 -> ~0.714
print("metric OK")
print("ALL UNIT TESTS PASSED")
