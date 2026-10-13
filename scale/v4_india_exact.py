"""Build india test npz with EXACT legs only (no TF-IDF) - OOM workaround.
Uses L2 (folded exact) + L3 (v3norm exact). Fast, no large matrix.
"""
import numpy as np, os, sys, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from common import log, NORM, read_chunks
from v4_block_aug import load_fold_dict, fold_name, build_name_maps, shard_s1_folded
from v3_block import v3_norm
from block import discover_shards

S1_PATH = os.path.join(NORM, "test_source1.tsv")
POOL_PATHS = [os.path.join(NORM, "test_source2.tsv"),
              os.path.join(NORM, "test_source3.tsv")]
OUT = "tmp/test_india_cand100.npz"
FILL_TO = 100

log("discovering shards")
SHARD_MAP, _ = discover_shards(POOL_PATHS)

log("loading fold dict")
tmap, dev_suffix = load_fold_dict()

log("getting india S1 folded names")
s1_leg = shard_s1_folded(S1_PATH, "india", SHARD_MAP, tmap, dev_suffix)
s1_ids = list(s1_leg.keys())
log(f"india S1: {len(s1_ids):,}")

want_folded = set(fn for fn, vn in s1_leg.values() if fn)
want_v3n = set(vn for fn, vn in s1_leg.values() if vn)
log(f"want folded={len(want_folded):,} v3n={len(want_v3n):,}")

log("building india pool name maps")
pool_ids, fold_map, v3n_map = build_name_maps(
    POOL_PATHS, "india", SHARD_MAP, tmap, dev_suffix, want_folded, want_v3n)
n_pool = len(pool_ids)
log(f"pool={n_pool:,}")

log("assembling exact candidates")
n = len(s1_ids)
top_rows = np.full((n, FILL_TO), -1, dtype=np.int32)
top_cos = np.zeros((n, FILL_TO), dtype=np.float32)
for i, eid in enumerate(s1_ids):
    fn, vn = s1_leg[eid]
    rows = []
    if fn:
        rows.extend(fold_map.get(fn, ()))
    if vn:
        rows.extend(v3n_map.get(vn, ()))
    # dedupe, cap
    seen = set()
    uniq = []
    for r in rows:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
            if len(uniq) >= FILL_TO:
                break
    k = len(uniq)
    if k:
        top_rows[i, :k] = uniq
        top_cos[i, :k] = 1.0
    if (i + 1) % 100000 == 0:
        log(f"  {i+1:,}/{n:,}")

mean_c = (top_rows >= 0).sum(axis=1).mean()
log(f"mean exact candidates per S1: {mean_c:.2f}")

log(f"writing {OUT}")
np.savez(OUT, s1_ids=np.array(s1_ids),
         top_rows=top_rows, top_cos=top_cos, pool_ids=pool_ids)
log("done")
