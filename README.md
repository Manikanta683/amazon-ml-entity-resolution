# Entity Resolution at Scale — Amazon ML Challenge 2026

**Team Byte Builders** · Public leaderboard: **0.850** (F_0.5)

Matches business entities in a source dataset against duplicates hidden in two much
larger target datasets (~2.2M labeled training pairs, ~10M+ inference records).

## Approach

1. **Blocking** (`scale/block.py`, `scale/v4_block_aug.py`) — per-country-shard
   word-unigram TF-IDF blocking. Finds 93% of true duplicate pairs while holding
   candidates to ~15 per entity (vs ~53 in the naive pool).
2. **Featurization** (`scale/featurize.py`) — pair-similarity features:
   rapidfuzz string scores, TF-IDF cosine, token-set ratios, name/address similarity.
3. **Classification** (`scale/train_model.py`, `scale/v4_train.py`) — gradient
   boosting on pair features. Validation F_0.5 **0.88** vs 0.81 for logistic
   regression; decision threshold tuned for F_0.5 (0.62).
4. **Inference** (`scale/score_gpu.py`) — chunked streaming writes; the full
   ~1.1 GB dataset processed in a single run.

## Results

| Stage | F_0.5 |
|---|---|
| Logistic regression baseline | 0.81 |
| Gradient boosting (tuned) | **0.88** (validation) |
| Public leaderboard | **0.850** |

## Layout

- `solution.py` — end-to-end reference pipeline (small-scale)
- `scale/` — production pipeline: blocking, featurization, training, scoring
- `make_synthetic.py` — synthetic data generator for local testing

## Run

```bash
pip install -r requirements.txt
python make_synthetic.py        # generate toy data
python solution.py              # run reference pipeline
```

Built by Manikanta Chowdavaram (Team Byte Builders) — B.Tech AI & ML, Parul University.
