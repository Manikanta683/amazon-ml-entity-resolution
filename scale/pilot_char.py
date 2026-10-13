#!/usr/bin/env python3
"""Pilot: char n-gram TF-IDF blocking recall on a tune sample.

Tests whether char 3-5 grams catch pairs that word-unigrams miss.
Uses india shard (weaker), 2000 tune S1s.
"""
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from common import NORM, TMP, ART, DATASET, read_chunks, log


def main():
    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    tune_ids = set(str(x) for x in sp["tune_ids"].tolist())
    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in tune_ids:
                gt[sid] = set(m for m in str(mids).split(",") if m and m != "nan")

    # load india tune S1 rows (sample 2000)
    rows = []
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
        m = chunk["entity_id"].isin(tune_ids) & (chunk["country_norm"] == "india")
        sub = chunk[m]
        rows.extend(zip(sub["entity_id"], sub["text"]))
        if len(rows) >= 2000:
            rows = rows[:2000]
            break
    log(f"sample S1: {len(rows)}")

    # load india pool (S2+S3)
    pool_ids, pool_texts = [], []
    for src in ("train_source2", "train_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
            m = chunk["country_norm"] == "india"
            sub = chunk[m]
            pool_ids.extend(sub["entity_id"].tolist())
            pool_texts.extend(sub["text"].tolist())
    log(f"pool: {len(pool_ids):,}")

    # char n-gram index
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=2, sublinear_tf=True, dtype=np.float32)
    P = vec.fit_transform(pool_texts)
    nn = NearestNeighbors(n_neighbors=60, metric="cosine")
    nn.fit(P)
    log("index built")

    # query and measure recall
    hit = tot = 0
    for a in range(0, len(rows), 500):
        b = min(len(rows), a + 500)
        Q = vec.transform([r[1] for r in rows[a:b]])
        dist, idx = nn.kneighbors(Q)
        for j in range(b - a):
            sid = rows[a + j][0]
            g = gt.get(sid, set())
            if not g:
                continue
            cands = set(pool_ids[i] for i in idx[j])
            hit += len(g & cands)
            tot += len(g)
    log(f"char n-gram top-60 recall: {hit}/{tot}={hit/max(tot,1):.4f}")
    with open(os.path.expanduser("~/workspace/amazon_ml/scale/pilot_char.txt"), "w") as f:
        f.write(f"recall={hit/max(tot,1):.4f}\n")


if __name__ == "__main__":
    main()
