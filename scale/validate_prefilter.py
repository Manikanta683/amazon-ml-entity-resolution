#!/usr/bin/env python3
"""Quick prefilter validation on 5k tune S1s: top-20 cosine + exact-name matches."""
import os, sys, time, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NORM, ART, TMP, log, read_chunks
from block import ShardIndex, discover_shards
from featurize import featurize_chunk
from train_model import load_pool_lists, tune_threshold
from common import macro_f05_from_counts

SAMPLE = 5000
shard = "india"  # test the harder shard

# load tune S1 ids for india from existing npz
d = np.load(os.path.join(TMP, f"tune_{shard}_cand.npz"), allow_pickle=True)
all_ids = [str(x) for x in d["s1_ids"].tolist()]
sample_ids = all_ids[:SAMPLE]
want = set(sample_ids)

# load GT for sample
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import load_gt_for_ids
gt = load_gt_for_ids(want)

# get S1 rows
rows = []
for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
    m = chunk["entity_id"].isin(want)
    sub = chunk[m]
    rows.extend(zip(sub["entity_id"], sub["name_norm"], sub["text"]))
    if len(rows) >= SAMPLE:
        break
rows = rows[:SAMPLE]
log(f"sample rows: {len(rows)}")

# build index (india shard only)
pool_paths = [os.path.join(NORM, "train_source2.tsv"), os.path.join(NORM, "train_source3.tsv")]
shard_map, shard_counts = discover_shards(pool_paths)
s1_names = {r[1] for r in rows if r[1]}
t0 = time.time()
index = ShardIndex(pool_paths, shard, shard_map, s1_names, n_docs=shard_counts[shard])
log(f"index built in {time.time()-t0:.0f}s")

# block with top-50, then prefilter to top-20 + exact name
t0 = time.time()
texts = [r[2] for r in rows]
names = [r[1] for r in rows]
cids, ccos = index.query(texts, names, topn=50, cap=50)
t_block = time.time() - t0
log(f"blocked {len(rows)} in {t_block:.1f}s ({t_block/len(rows)*1000:.1f} ms/S1)")

# prefilter: top-20 by cosine + all exact name matches (already in cids via fallback)
# For validation: take first 20 of the 50 (they're cosine-sorted)
pref_cids = [c[:20] for c in cids]
pref_cos = [c[:20] for c in ccos]

# recall after prefilter
n_gt = rec = 0
for sid, pc in zip([r[0] for r in rows], pref_cids):
    g = gt.get(sid, set())
    if g:
        n_gt += 1
        rec += len(g & set(pc)) / len(g)
log(f"prefilter recall: {rec/max(1,n_gt):.4f} (vs blocking 0.9324)")

# featurize top-20 pairs
pool_names, pool_addrs, pool_ctry = load_pool_lists(pool_paths, shard=shard, shard_map=shard_map)
id2row = {e.decode(): r for r, e in enumerate(index.ids)}
with open(os.path.join(ART, "model.pkl"), "rb") as f:
    model = pickle.load(f)
thr = float(open(os.path.join(ART, "threshold.txt")).read().strip())

t0 = time.time()
s1_recs, rows_list, cos_list = [], [], []
for (sid, nm, _), pc, pcos in zip(rows, pref_cids, pref_cos):
    # get S1 fields
    pass  # simplified: use name from rows

# need S1 address/country; re-read
fields = {}
for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
    m = chunk["entity_id"].isin(want)
    sub = chunk[m]
    for eid, nm, ad, ct in zip(sub["entity_id"], sub["name_norm"], sub["address_norm"], sub["country_norm"]):
        fields[eid] = (nm, ad, ct)
    if len(fields) >= len(want):
        break

for (sid, _, _), pc, pcos in zip(rows, pref_cids, pref_cos):
    nm, ad, ct = fields[sid]
    prows = [id2row[c] for c in pc]
    s1_recs.append((nm, ad, ct))
    rows_list.append(prows)
    cos_list.append(pcos[:len(prows)])

X, _ = featurize_chunk(s1_recs, rows_list, cos_list, pool_names, pool_addrs, pool_ctry)
t_feat = time.time() - t0
log(f"featurized {len(X)} pairs in {t_feat:.1f}s ({t_feat/len(X)*1e6:.0f} us/pair)")

# predict and score
t0 = time.time()
prob = model.predict_proba(X)[:, 1]
t_pred = time.time() - t0
log(f"predicted in {t_pred:.1f}s ({t_pred/len(X)*1e6:.0f} us/pair)")

# build per-entity predictions
n_ent = len(rows)
ent_idx = np.repeat(np.arange(n_ent), [len(r) for r in rows_list])
y_true = np.zeros(len(X), dtype=np.int8)
k = 0
for i, (sid, prows) in enumerate(zip([r[0] for r in rows], rows_list)):
    g = gt.get(sid, set())
    for t, r in enumerate(prows):
        eid = index.ids[r].decode()
        if eid in g:
            y_true[k] = 1
        k += 1
gt_counts = np.array([len(gt.get(r[0], set())) for r in rows], dtype=np.int32)

bt, bf, _, _ = tune_threshold(y_true, prob, ent_idx, gt_counts, n_ent)
log(f"PREFILTER tune F05={bf:.4f} thr={bt:.3f} (vs baseline 0.8805)")

# also with global threshold
pred = prob >= thr
tp = np.bincount(ent_idx[pred & (y_true==1)], minlength=n_ent)
fp = np.bincount(ent_idx[pred & (y_true==0)], minlength=n_ent)
fn = gt_counts - tp
f_global = macro_f05_from_counts(tp, fp, fn)
log(f"PREFILTER F05 @global thr {thr:.3f}: {f_global:.4f}")
