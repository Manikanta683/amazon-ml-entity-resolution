#!/usr/bin/env python3
"""v3 blocking micro-test (small pool): validates ShardIndexV3 two-stage
recall gain vs stage-1 only, and measures per-query timing.

Builds a small pool (GT pool rows for N tune S1 + random filler), fits the
word + char-bigram indices, and compares pair recall at top-15 vs top-100
(two-stage). Temp TSVs live in /tmp (ephemeral test only).
"""
import os
import sys
import time
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from common import NORM, TMP, ART, DATASET, read_chunks, log  # noqa: E402
from block import discover_shards  # noqa: E402
from v3_block import ShardIndexV3, query_two_stage  # noqa: E402

N_S1 = 2000
FILLER = 48000


def main():
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    tune_ids = [str(x) for x in sp["tune_ids"].tolist()]
    d = np.load(os.path.join(TMP, "tune_india_cand.npz"), allow_pickle=True)
    s1_india = [str(x) for x in d["s1_ids"].tolist()]
    rng = np.random.RandomState(1)
    s1 = [s1_india[i] for i in rng.choice(len(s1_india), N_S1, replace=False)]
    s1set = set(s1)

    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in s1set:
                gt[sid] = set(m for m in str(mids).split(",") if m and m != "nan")
    gt_pids = set(p for ps in gt.values() for p in ps)
    log(f"S1={len(s1):,} GT pool ids={len(gt_pids):,}")

    # collect filler pool rows (india shard) + all GT pool rows
    filler, s1rows = [], {}
    need_filler = FILLER
    for src in ("train_source2", "train_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
            m = chunk["country_norm"] == "india"
            sub = chunk[m]
            for eid, nm, ad, cy, tx in zip(sub["entity_id"], sub["name_norm"],
                                           sub["address_norm"], sub["country_norm"],
                                           sub["text"]):
                if eid in gt_pids:
                    filler.append((eid, nm, ad, cy, tx))
                    gt_pids.discard(eid)
                elif need_filler > 0 and rng.random() < 0.05:
                    filler.append((eid, nm, ad, cy, tx))
                    need_filler -= 1
            if need_filler <= 0 and not gt_pids:
                break
        if need_filler <= 0 and not gt_pids:
            break
    log(f"pool rows={len(filler):,} (missing GT: {len(gt_pids)})")
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
        m = chunk["entity_id"].isin(s1set)
        for eid, nm, ad, cy, tx in zip(
                chunk.loc[m, "entity_id"], chunk.loc[m, "name_norm"],
                chunk.loc[m, "address_norm"], chunk.loc[m, "country_norm"],
                chunk.loc[m, "text"]):
            s1rows[eid] = (nm, ad, cy, tx)

    tmpd = tempfile.mkdtemp(prefix="v3block_")
    pool_p = os.path.join(tmpd, "pool.tsv")
    s1_p = os.path.join(tmpd, "s1.tsv")
    pd.DataFrame(filler, columns=["entity_id", "name_norm", "address_norm",
                                  "country_norm", "text"]).to_csv(
        pool_p, sep="\t", index=False)
    pd.DataFrame([(e, *s1rows[e]) for e in s1],
                 columns=["entity_id", "name_norm", "address_norm",
                          "country_norm", "text"]).to_csv(
        s1_p, sep="\t", index=False)

    shard_map, _ = discover_shards([pool_p])
    s1_names_all = set(r[1] for r in filler)
    idx = ShardIndexV3([pool_p], "india", shard_map, s1_names_all)
    idx.build_v3norm_index([pool_p], "india", shard_map)

    texts = [s1rows[e][3] for e in s1]
    names = [s1rows[e][0] for e in s1]
    pid2row = {r[0]: i for i, r in enumerate(filler)}

    # stage-1 only (top-15)
    t0 = time.time()
    rows15, cos15 = idx.query_rows_stage1(texts, names, stage1_n=15)
    t1 = time.time()
    rows100, cos100 = query_two_stage(idx, texts, names)
    t2 = time.time()

    def recall(rows_list):
        hit = tot = 0
        for e, rows in zip(s1, rows_list):
            cand = set(filler[r][0] for r in rows)
            for p in gt.get(e, ()):  # noqa: B007
                tot += 1
                if p in cand:
                    hit += 1
        return hit / max(tot, 1), hit, tot

    r15 = recall(rows15)
    r100 = recall(rows100)
    log(f"stage1 top-15:  recall={r15[0]:.4f} ({r15[1]:,}/{r15[2]:,}) "
        f"{(t1-t0)/len(s1)*1000:.1f} ms/S1")
    log(f"two-stage 100:  recall={r100[0]:.4f} ({r100[1]:,}/{r100[2]:,}) "
        f"{(t2-t1)/len(s1)*1000:.1f} ms/S1")
    log(f"mean cands: {np.mean([len(r) for r in rows100]):.1f}")


if __name__ == "__main__":
    main()
