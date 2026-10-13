#!/usr/bin/env python3
"""Full test-set prediction.

Phase A (per shard): block test S1s -> TMP/test_{shard}_cand.npz
Phase B (per shard): stage-2 features + classifier -> TMP/test_{shard}_matches.tsv
Merge: stream test S1 in file order -> output_real/matching_results.tsv +
       output_real/candidate_pairs.tsv

Usage: python3 run_test.py [--limit-per-shard N] [--outdir DIR]
"""
import argparse
import gc
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    NORM, ART, TMP, OUT_REAL, read_chunks, log,
)
from block import discover_shards, ShardIndex  # noqa: E402
from featurize import featurize_chunk  # noqa: E402
from train_model import load_pool_lists  # noqa: E402

STAGE1_TOP = 15
QCHUNK = 1000
FCHUNK = 10_000
CAND_CAP = 60


def phase_a(shard, shard_map, shard_counts, limit=None):
    out_path = os.path.join(TMP, f"test_{shard}_cand.npz")
    if os.path.exists(out_path):
        log(f"[{shard}] cached {out_path}")
        return out_path
    t0 = time.time()
    pool_paths = [os.path.join(NORM, "test_source2.tsv"),
                  os.path.join(NORM, "test_source3.tsv")]
    rows = []  # (id, name, text)
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv")):
        m = chunk["country_norm"].map(lambda c: shard_map.get(c, "other")) == shard
        sub = chunk[m]
        rows.extend(zip(sub["entity_id"], sub["name_norm"], sub["text"]))
        if limit and len(rows) >= limit:
            rows = rows[:limit]
            break
    log(f"[{shard}] S1 rows to block: {len(rows):,}")
    s1_names = {r[1] for r in rows if r[1]}
    index = ShardIndex(pool_paths, shard, shard_map, s1_names,
                       n_docs=shard_counts.get(shard))
    id2row = {e.decode(): r for r, e in enumerate(index.ids)}

    n = len(rows)
    cand60 = np.full((n, CAND_CAP), -1, dtype=np.int32)
    top_rows = np.full((n, STAGE1_TOP), -1, dtype=np.int32)
    top_cos = np.zeros((n, STAGE1_TOP), dtype=np.float32)
    for a in range(0, n, QCHUNK):
        b = min(n, a + QCHUNK)
        texts = [r[2] for r in rows[a:b]]
        names = [r[1] for r in rows[a:b]]
        cand_ids, cand_cos = index.query(texts, names)
        for j in range(b - a):
            cids = cand_ids[j]
            for t, cid in enumerate(cids[:CAND_CAP]):
                cand60[a + j, t] = id2row[cid]
            k = min(STAGE1_TOP, len(cids))
            for t in range(k):
                top_rows[a + j, t] = id2row[cids[t]]
                top_cos[a + j, t] = cand_cos[j][t]
        if (a // QCHUNK) % 20 == 0:
            log(f"[{shard}] blocked {b:,}/{n:,}")
    np.savez_compressed(out_path, s1_ids=np.array([r[0] for r in rows]),
                        cand60=cand60, top_rows=top_rows, top_cos=top_cos,
                        pool_ids=index.ids)
    log(f"[{shard}] phase A done in {time.time() - t0:.0f}s -> {out_path}")
    del index, id2row
    gc.collect()
    return out_path


def phase_b(shard, shard_map, model, thr):
    out_tsv = os.path.join(TMP, f"test_{shard}_matches.tsv")
    if os.path.exists(out_tsv):
        log(f"[{shard}] cached {out_tsv}")
        return out_tsv
    t0 = time.time()
    d = np.load(os.path.join(TMP, f"test_{shard}_cand.npz"), allow_pickle=True)
    s1_ids = [str(x) for x in d["s1_ids"].tolist()]
    top_rows, top_cos, pool_ids = d["top_rows"], d["top_cos"], d["pool_ids"]
    n = len(s1_ids)
    pool_paths = [os.path.join(NORM, "test_source2.tsv"),
                  os.path.join(NORM, "test_source3.tsv")]
    pool_names, pool_addrs, pool_ctry = load_pool_lists(
        pool_paths, shard=shard, shard_map=shard_map)
    # S1 fields, aligned by position
    fields = {}
    want = set(s1_ids)
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv")):
        m = chunk["entity_id"].isin(want)
        sub = chunk[m]
        for eid, nm, ad, ct in zip(sub["entity_id"], sub["name_norm"],
                                   sub["address_norm"], sub["country_norm"]):
            fields[eid] = (nm, ad, ct)
        if len(fields) >= len(want):
            break
    log(f"[{shard}] S1 fields loaded: {len(fields):,}")
    with open(out_tsv, "w") as f:
        for a in range(0, n, FCHUNK):
            b = min(n, a + FCHUNK)
            s1_recs, rows_list, cos_list, pos_list = [], [], [], []
            for j in range(a, b):
                rows = [int(r) for r in top_rows[j] if r >= 0]
                if not rows:
                    continue
                nm, ad, ct = fields[s1_ids[j]]
                s1_recs.append((nm, ad, ct))
                rows_list.append(rows)
                cos_list.append([float(top_cos[j, t]) for t in range(len(rows))])
                pos_list.append(j)
            matched = {}
            if s1_recs:
                Xc, p_s1 = featurize_chunk(s1_recs, rows_list, cos_list,
                                           pool_names, pool_addrs, pool_ctry)
                prob = model.predict_proba(Xc)[:, 1]
                # per-S1: ids above threshold, prob-desc
                k = 0
                for pi, (rows, pos) in enumerate(zip(rows_list, pos_list)):
                    scored = [(float(prob[k + t]), pool_ids[rows[t]].decode())
                              for t in range(len(rows))]
                    k += len(rows)
                    scored.sort(key=lambda x: -x[0])
                    matched[pos] = ",".join(eid for p, eid in scored if p >= thr)
            for j in range(a, b):
                f.write(f"{s1_ids[j]}\t{matched.get(j, '')}\n")
            log(f"[{shard}] predicted {b:,}/{n:,}")
    log(f"[{shard}] phase B done in {time.time() - t0:.0f}s")
    del pool_names, pool_addrs, pool_ctry
    gc.collect()
    return out_tsv


def merge(shards, shard_map, outdir):
    t0 = time.time()
    # per-shard stores
    stores = {}
    match_of = {}
    for shard in shards:
        d = np.load(os.path.join(TMP, f"test_{shard}_cand.npz"), allow_pickle=True)
        s1_ids = [str(x) for x in d["s1_ids"].tolist()]
        stores[shard] = {
            "pos": {sid: i for i, sid in enumerate(s1_ids)},
            "cand60": d["cand60"],
            "pool_ids": d["pool_ids"],
            "s1_ids": s1_ids,
        }
        md = {}
        with open(os.path.join(TMP, f"test_{shard}_matches.tsv")) as f:
            for line in f:
                sid, _, mids = line.rstrip("\n").partition("\t")
                md[sid] = mids
        match_of[shard] = md
        log(f"[{shard}] merge stores ready: {len(s1_ids):,} S1")
    os.makedirs(outdir, exist_ok=True)
    mp = os.path.join(outdir, "matching_results.tsv")
    cp = os.path.join(outdir, "candidate_pairs.tsv")
    n_out = 0
    with open(mp, "w") as fm, open(cp, "w") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv"),
                                 usecols=["entity_id", "country_norm"]):
            ml, cl = [], []
            for sid, ctry in zip(chunk["entity_id"], chunk["country_norm"]):
                shard = shard_map.get(ctry, "other")
                st = stores.get(shard)
                if st is None or sid not in st["pos"]:
                    continue
                p = st["pos"][sid]
                rows = st["cand60"][p]
                cids = [st["pool_ids"][r].decode() for r in rows if r >= 0]
                ml.append(f"{sid}\t{match_of[shard].get(sid, '')}\n")
                cl.append(f"{sid}\t{','.join(cids)}\n")
                n_out += 1
            fm.writelines(ml)
            fc.writelines(cl)
    log(f"merge done: {n_out:,} rows -> {outdir} ({time.time()-t0:.0f}s)")
    return mp, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-shard", type=int, default=None)
    ap.add_argument("--outdir", default=OUT_REAL)
    args = ap.parse_args()
    t_all = time.time()

    pool_paths = [os.path.join(NORM, "test_source2.tsv"),
                  os.path.join(NORM, "test_source3.tsv")]
    shard_map, shard_counts = discover_shards(pool_paths)
    # shards present in test S1
    seen = set()
    for chunk in read_chunks(os.path.join(NORM, "test_source1.tsv"),
                             usecols=["country_norm"]):
        seen.update(chunk["country_norm"].map(
            lambda c: shard_map.get(c, "other")).tolist())
    shards = sorted(seen)
    log(f"test shards: {shards}")

    with open(os.path.join(ART, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    thr = float(open(os.path.join(ART, "threshold.txt")).read().strip())
    log(f"model loaded; threshold={thr:.4f}")

    for shard in shards:
        phase_a(shard, shard_map, shard_counts, limit=args.limit_per_shard)
    for shard in shards:
        phase_b(shard, shard_map, model, thr)
    mp, cp = merge(shards, shard_map, args.outdir)
    log(f"TOTAL {(time.time() - t_all) / 60:.1f} min\n  {mp}\n  {cp}")


if __name__ == "__main__":
    main()
