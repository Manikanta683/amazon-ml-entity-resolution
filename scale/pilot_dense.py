#!/usr/bin/env python3
"""Pilot: dense blocking recall with all-MiniLM-L6-v2 on a tune sample.

Embeds 2000 tune S1 queries + india pool sample, brute-force cosine,
measures recall. Tests if dense catches TF-IDF misses.
"""
import os
import numpy as np
from common import NORM, TMP, ART, DATASET, read_chunks, log


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    log("model loaded")

    sp = np.load(os.path.join(ART, "samples.npz"), allow_pickle=True)
    tune_ids = set(str(x) for x in sp["tune_ids"].tolist())
    gt = {}
    for chunk in read_chunks(os.path.join(DATASET, "train", "train_ground_truth.tsv")):
        for sid, mids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if sid in tune_ids:
                gt[sid] = set(m for m in str(mids).split(",") if m and m != "nan")

    # 2000 india tune S1
    rows = []
    for chunk in read_chunks(os.path.join(NORM, "train_source1.tsv")):
        m = chunk["entity_id"].isin(tune_ids) & (chunk["country_norm"] == "india")
        sub = chunk[m]
        rows.extend(zip(sub["entity_id"], sub["business_name"] + " " + sub["business_address"]))
        if len(rows) >= 2000:
            rows = rows[:2000]
            break

    # india pool sample: 100k (for pilot speed)
    pool_ids, pool_texts = [], []
    for src in ("train_source2", "train_source3"):
        for chunk in read_chunks(os.path.join(NORM, f"{src}.tsv")):
            m = chunk["country_norm"] == "india"
            sub = chunk[m]
            pool_ids.extend(sub["entity_id"].tolist())
            pool_texts.extend((sub["business_name"] + " " + sub["business_address"]).tolist())
            if len(pool_ids) >= 100000:
                break
        if len(pool_ids) >= 100000:
            break
    pool_ids, pool_texts = pool_ids[:100000], pool_texts[:100000]
    log(f"queries={len(rows)}, pool={len(pool_ids)}")

    # embed
    q_emb = model.encode([r[1] for r in rows], batch_size=64,
                         show_progress_bar=False, convert_to_numpy=True)
    p_emb = model.encode(pool_texts, batch_size=64,
                         show_progress_bar=False, convert_to_numpy=True)
    log("embedded")

    # brute-force top-60 cosine
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    p_emb = p_emb / np.linalg.norm(p_emb, axis=1, keepdims=True)
    hit = tot = 0
    for j in range(0, len(rows), 200):
        b = min(len(rows), j + 200)
        sims = (q_emb[j:b] @ p_emb.T)
        topk = np.argpartition(-sims, 60, axis=1)[:, :60]
        for k in range(b - j):
            sid = rows[j + k][0]
            g = gt.get(sid, set())
            if not g:
                continue
            cands = set(pool_ids[i] for i in topk[k])
            # only count GT pairs that are in our 100k pool sample
            g_in = g & set(pool_ids)
            hit += len(g_in & cands)
            tot += len(g_in)
    rec = hit / max(tot, 1)
    log(f"dense top-60 recall (100k pool): {hit}/{tot}={rec:.4f}")
    with open(os.path.expanduser("~/workspace/amazon_ml/scale/pilot_dense.txt"), "w") as f:
        f.write(f"recall={rec:.4f}\n")


if __name__ == "__main__":
    main()
