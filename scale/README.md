# Scale pipeline — Amazon ML Challenge 2026 (entity resolution)

Two-stage blocking + classifier, sharded by country so test-only countries
(France, …) never hit a hardcoded label set.

## Files

| script | what it does |
|---|---|
| `common.py` | normalization, chunked TSV IO, macro F_0.5, threshold picker |
| `preprocess.py` | raw TSV → `norm/*.tsv` (normalized name/address/country/text) |
| `block.py` | `discover_shards`, `ShardIndex` (TF-IDF char/word blocking + exact-name fallback) |
| `featurize.py` | 16 pair-similarity features (RapidFuzz, parallel) |
| `train_model.py` | sample 8k/1.5k S1 → block → featurize → train LR vs HGB → tune threshold for macro F_0.5 → `artifacts/model.pkl`, `artifacts/threshold.txt` |
| `run_test.py` | phase A: block all test S1s → `tmp/test_{shard}_cand.npz`; phase B: stage-2 classify → per-shard match TSVs; merge in test file order → `output_real/*.tsv` |
| `check_format.py` | streaming format gate (headers, order, prefixes, dupes, matched ⊆ candidates, optional ID existence) |
| `test_unit.py` | unit tests for block/featurize/threshold/metric |

## Run order

```
python3 preprocess.py
python3 train_model.py            # writes artifacts/model.pkl, artifacts/threshold.txt
python3 run_test.py               # writes output_real/matching_results.tsv + candidate_pairs.tsv
python3 check_format.py --matching output_real/matching_results.tsv \
    --candidate output_real/candidate_pairs.tsv [--check-ids]
```

Dry run (2k S1s/shard, exercises the full path including the real TF-IDF build):

```
python3 run_test.py --limit-per-shard 2000 --outdir dryrun_out
tail -n +2 dryrun_out/matching_results.tsv | cut -f1 > /tmp/dryrun_order.txt
python3 check_format.py --matching dryrun_out/matching_results.tsv \
    --candidate dryrun_out/candidate_pairs.tsv --s1-order /tmp/dryrun_order.txt
```

## Design notes

- **Sharding**: country labels counted on the pool; labels with <50k docs (or
  empty) → `other` shard. Test-only countries land in `other` automatically.
- **Blocking**: per-shard TF-IDF (word 2+ chars, min_df=2, max_df=1%,
  sublinear) cosine top-60; exact-name fallback map tops up to 60.
  Stage-1 keeps top-15 by cosine for the classifier.
- **Features (16)**: name/addr RapidFuzz ratios, token overlap, cosine,
  country match, length features, name-equality flags.
- **Model**: LogisticRegression (balanced) vs HistGradientBoosting, picked by
  tune-set macro F_0.5; per-entity threshold swept 0.02–0.98, mid-plateau of
  the best score (precision-leaning, good for F_0.5).
- **Outputs**: `matching_results.tsv` predicted matches (prob-desc);
  `candidate_pairs.tsv` all ≤60 blocked candidates; both in test S1 file
  order, one row per S1, empty = singleton.
- **Memory**: 7 GB box. Pool string columns are de-duplicated per shard;
  train matrices go to disk-backed memmaps; format check streams.
- Deterministic: `SEED=20260925` everywhere (sampling, sklearn, joblib).

## Artifacts

- `norm/` — normalized TSVs (regenerable)
- `tmp/` — per-shard candidate npz, match TSVs, memmaps (scratch)
- `artifacts/` — `model.pkl`, `threshold.txt`, `report.txt`, `samples.npz`, `shard_map.txt`
- `output_real/` — final submission files

## v3 (top-10 push)

New scripts (all features from provided TSVs only):

| script | what it does |
|---|---|
| `v3_featurize.py` | 30 pair features: v2's 20 + name/addr-only TF-IDF cosines, WRatio, char-3gram Jaccard/containment, consonant-skeleton Jaccard (transliteration), first-token match, digit Jaccard, v3-norm token containment |
| `v3_block.py` | `ShardIndexV3`: dual word-TF-IDF indices (`text` + v3-normalized text: diacritic-fold + double-letter collapse); two-stage query top-15 + 50 = 100 |
| `v3_block_test_small.py` | micro-test: 50k-pool recall@15 vs recall@100 + timing |
| `v3_train.py` | `--sample N` validation or full train; HGB (+LR sweep, dropped: adds nothing); global threshold tuned on **correct** macro F_0.5 (fn includes blocking misses); saves `artifacts_v3/` |
| `v3_block_full.py` | HEAVY: rebuild test blocking top-100 -> `tmp/test_{shard}_cand100.npz` + name/addr vec pickles (run after v2 finishes) |
| `v3_test.py` | HEAVY: two-pass test scoring -> `output_v3/` (refuses to run while v2_min.py alive) |
| `v3_run_full.sh` | full launch sequence: block -> train -> test -> validate -> upload copy |

Key validation findings (3k/1.5k S1 sample, correct metric):
- 30 features beat 20 features on identical data (v3 vs v2-style baseline)
- Scaled LR ensemble: weight 1.0 (HGB-only) wins -> LR dropped
- Per-entity rules ((beta,lo) grid; per-entity expected-F_0.5 argmax) tie/lose vs global threshold -> global threshold shipped
- No safe cheap prefilter exists on tune (hard positives fail set-op rules) -> two-pass uncertainty gate instead (pass-1 top-15 for all; stage-2 extras only for S1s with max prob < 0.95 or <2 confident; ~22% of S1s)
- Metric caution: v2's 0.9148 ignored blocking-miss FNs (optimistic ~0.02-0.03); v3 tunes on the correct metric
