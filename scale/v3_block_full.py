#!/usr/bin/env python3
"""v3 full blocking rebuild for TEST shards (HEAVY - run only after v2_min.py
finishes; it builds two TF-IDF indices per shard on 1.4-4.7M pool docs).

Per shard in (france, india, us[, other]):
  1. discover shards on test pool
  2. build ShardIndexV3 (word index) + build_v3norm_index (sequentially, to
     bound peak RAM: query with word index, then free it before v3norm? No -
     query_two_stage needs both. Peak = word idx + v3norm idx ~ 2x v2.)
  3. query all test S1 in 10k batches -> tmp/test_{shard}_cand100.npz with
     s1_ids, top_rows (n,100) int32, top_cos (n,100) float32, pool_ids
  4. fit name/addr TF-IDF vectorizers on the shard pool -> artifacts_v3/
     vec_name_{shard}.pkl, vec_addr_{shard}.pkl (vocab+idf only, small)

Memory strategy: process ONE shard at a time; after querying, drop the
indices before the next shard. The v3norm texts are generated on the fly
(v3_norm applied in doc_iter), no preprocess rerun needed.
"""
import os
import sys
import gc
import pickle
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, ART, read_chunks, log  # noqa: E402
from block import discover_shards, collect_s1_names  # noqa: E402
from v3_block import ShardIndexV3, query_two_stage  # noqa: E402
from v3_featurize import v3_norm  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

V3_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_v3")
BATCH = 1_000  # 500 too slow (10h); 2000 OOMs; trying 1000


def fit_name_addr_vecs(pool_paths, shard, shard_map):
    kw = dict(analyzer="word", token_pattern=r"(?u)\b\w\w+\b",
              min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)

    def name_iter():
        for path in pool_paths:
            for chunk in read_chunks(path, usecols=["country_norm", "name_norm"]):
                m = (chunk["country_norm"].map(
                    lambda c: shard_map.get(c, "other")) == shard)
                yield from chunk.loc[m, "name_norm"].tolist()

    def addr_iter():
        for path in pool_paths:
            for chunk in read_chunks(path, usecols=["country_norm", "address_norm"]):
                m = (chunk["country_norm"].map(
                    lambda c: shard_map.get(c, "other")) == shard)
                yield from chunk.loc[m, "address_norm"].tolist()

    log(f"[{shard}] fitting name TF-IDF ...")
    vn = TfidfVectorizer(**kw).fit(name_iter())
    log(f"[{shard}] fitting addr TF-IDF ...")
    va = TfidfVectorizer(**kw).fit(addr_iter())
    return vn, va


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--stage1-only", action="store_true",
                    help="skip v3norm index (OOM-safe for big shards); "
                         "stage-1 top-15 + exact-name extras only")
    args = ap.parse_args()

    pool_paths = [os.path.join(NORM, "test_source2.tsv"),
                  os.path.join(NORM, "test_source3.tsv")]
    s1_path = os.path.join(NORM, "test_source1.tsv")
    shard_map, shard_counts = discover_shards(pool_paths)
    shards = args.shards or sorted(set(shard_map.values()))
    log(f"shards to process: {shards}")

    os.makedirs(V3_ART, exist_ok=True)
    for shard in shards:
        t0 = time.time()
        n_docs = shard_counts.get(shard, 0)
        s1_names, n_s1 = collect_s1_names([s1_path], shard, shard_map)
        idx = ShardIndexV3(pool_paths, shard, shard_map, s1_names,
                           n_docs=n_docs)
        if args.stage1_only:
            log(f"[{shard}] stage1-only mode: skipping v3norm index "
                f"(OOM-safe)")
        else:
            idx.build_v3norm_index(pool_paths, shard, shard_map, n_docs=n_docs)

        # stream S1 in batches, query, accumulate
        all_rows, all_cos, s1_ids = [], [], []
        qb = []
        qt = []
        qn = []

        def flush():
            if not qb:
                return
            if args.stage1_only:
                rows, cos = idx.query_rows_stage1(qt, qn)
            else:
                rows, cos = query_two_stage(idx, qt, qn)
            for sid, r, c in zip(qb, rows, cos):
                s1_ids.append(sid)
                rr = np.full(100, -1, dtype=np.int32)
                cc = np.zeros(100, dtype=np.float32)
                rr[:len(r)] = r
                cc[:len(c)] = c
                all_rows.append(rr)
                all_cos.append(cc)
            qb.clear(); qt.clear(); qn.clear()

        for chunk in read_chunks(s1_path):
            m = (chunk["country_norm"].map(
                lambda c: shard_map.get(c, "other")) == shard)
            sub = chunk[m]
            for sid, tx, nm in zip(sub["entity_id"], sub["text"],
                                   sub["name_norm"]):
                qb.append(sid); qt.append(tx); qn.append(nm)
                if len(qb) >= BATCH:
                    flush()
                    log(f"[{shard}] queried {len(s1_ids):,}/{n_s1:,}")
        flush()
        log(f"[{shard}] queried {len(s1_ids):,} S1 in {time.time()-t0:.0f}s")

        top_rows = np.vstack(all_rows)
        top_cos = np.vstack(all_cos)
        pool_ids = np.array(idx.ids)  # already S16 bytes; no .encode() (np.bytes_ has none)
        out = os.path.join(TMP, f"test_{shard}_cand100.npz")
        np.savez(out, s1_ids=np.array(s1_ids),
                 top_rows=top_rows, top_cos=top_cos, pool_ids=pool_ids)
        log(f"[{shard}] saved {out} rows={top_rows.shape}")
        mean_c = (top_rows >= 0).sum(axis=1).mean()
        log(f"[{shard}] mean candidates/S1: {mean_c:.1f}")
        del all_rows, all_cos, top_rows, top_cos

        # name/addr vectorizers (small pickles)
        vn, va = fit_name_addr_vecs(pool_paths, shard, shard_map)
        pickle.dump(vn, open(os.path.join(V3_ART, f"vec_name_{shard}.pkl"), "wb"))
        pickle.dump(va, open(os.path.join(V3_ART, f"vec_addr_{shard}.pkl"), "wb"))

        del idx, vn, va
        gc.collect()
        log(f"[{shard}] done in {(time.time()-t0)/60:.0f} min")
    log("ALL SHARDS DONE")


if __name__ == "__main__":
    main()
