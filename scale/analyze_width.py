#!/usr/bin/env python3
"""Measure oracle F_0.5 ceiling at wider candidate widths (top-30/60).

Re-runs blocking queries on tune S1s, captures full candidate lists,
computes oracle (perfect classifier) F_0.5 at various top-K cutoffs.
This tells us the score ceiling vs candidate-size tradeoff.
Writes to ~/workspace (persistent).
"""
import os
import numpy as np
from common import NORM, TMP, ART, DATASET, read_chunks, log
from block import ShardIndex, discover_shards


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
    return 1.25 * prec * rec / (0.25 * prec + rec)


def main():
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    tune_ids = set(str(x) for x in sp["tune_ids"].tolist())
    gt = load_gt_for_ids(tune_ids)

    pool_paths = [os.path.join(NORM, "train_source2.tsv"),
                  os.path.join(NORM, "train_source3.tsv")]
    shard_map, shard_counts = discover_shards(pool_paths)

    out_lines = []
    for shard in ("india", "us"):
        # load tune S1 rows for this shard
        rows = []
        for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
            m = chunk["entity_id"].isin(tune_ids) & (
                chunk["country_norm"].map(lambda c: shard_map.get(c, "other")) == shard)
            sub = chunk[m]
            rows.extend(zip(sub["entity_id"], sub["name_norm"], sub["text"]))
        log(f"[{shard}] tune S1: {len(rows):,}")
        s1_names = {r[1] for r in rows if r[1]}
        index = ShardIndex(pool_paths, shard, shard_map, s1_names,
                           n_docs=shard_counts.get(shard))

        # accumulate oracle scores at different K
        K_vals = [15, 30, 45, 60]
        sums = {k: 0.0 for k in K_vals}
        n = 0
        for a in range(0, len(rows), 1000):
            b = min(len(rows), a + 1000)
            texts = [r[2] for r in rows[a:b]]
            names = [r[1] for r in rows[a:b]]
            cand_ids, _ = index.query(texts, names)  # entity IDs, cosine-desc
            for j in range(b - a):
                sid = rows[a + j][0]
                truth = gt.get(sid, set())
                cids = cand_ids[j]
                for k in K_vals:
                    pred = truth & set(cids[:k])
                    sums[k] += f05_per_entity(pred, truth)
                n += 1
            if (a // 1000) % 10 == 0:
                log(f"[{shard}] queried {b:,}/{len(rows):,}")
        for k in K_vals:
            line = f"{shard} top-{k}: oracle_F05={sums[k]/n:.4f}"
            log(line)
            out_lines.append(line)
        del index
        import gc; gc.collect()

    with open(os.path.expanduser("~/workspace/amazon_ml/scale/analysis_width.txt"), "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
