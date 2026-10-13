# v4 Workflow — Run on 16GB RAM machine

## What this is
The full v4 entity-resolution pipeline for Amazon ML Challenge 2026.
On 16GB RAM, the TF-IDF blocking works (no OOM). Expected score: 0.90-0.93.

## Setup
1. Python 3.10+ with: numpy, pandas, scikit-learn, scipy, rapidfuzz
   `pip install numpy pandas scikit-learn scipy rapidfuzz`
2. Dataset: `student_resource/` with `dataset/` inside (the 1.09GB zip from Sep 25).
   Place it so the path is: `./student_resource/student_resource/dataset/`

## Run (in order)
```bash
# 1. Mine transliteration dict (5 min)
python3 v4_mine.py

# 2. Test blocking — all 3 shards (TF-IDF, ~2 hours on 16GB)
python3 v3_block_full.py --shards france --stage1-only
python3 v3_block_full.py --shards india --stage1-only
python3 v3_block_full.py --shards us --stage1-only

# 3. Augment with exact legs (30 min)
python3 v4_block_aug.py --augment-test

# 4. Train full model (3 hours)
python3 v4_train.py --full --pos-k 0 --n-train 60000 --n-tune 12000

# 5. Test / predict (2.5 hours)
python3 v4_test.py

# Output: output_v4/matching_results.tsv (upload this)
```

## Files
- `v4_mine.py` — mines Devanagari→Latin mappings
- `v3_block_full.py` — TF-IDF blocking (stage1-only)
- `v4_block_aug.py` — adds exact-match legs
- `v4_train.py` — trains HGB classifier
- `v4_test.py` — scores test, writes submission
- `v4_featurize.py`, `common.py`, `block.py`, `v3_block.py` — shared

## Notes
- Total time: ~8 hours on 16GB.
- Output `matching_results.tsv` goes in `output_v4/`.
- Upload that file to the challenge portal.
