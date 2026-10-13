#!/bin/bash
# v3 FULL RUN - launch only AFTER v2_min.py has finished and its output is
# validated + copied for upload. Estimated total: ~8-12h on this 2-core/7GB box.
#
#   Step 1: v3_block_full.py  - dual-index blocking, top-100 test candidates
#   Step 2: v3_train.py       - 30 features, HGB+LR ensemble, adaptive threshold
#   Step 3: v3_test.py        - two-pass test scoring -> output_v3/
#   Step 4: validate + refresh upload copies
set -u
cd ~/workspace/amazon_ml/scale
LOGDIR=logs
mkdir -p "$LOGDIR" "$HOME/workspace/amazon_ml/output_v3"

if pgrep -f v2_min.py > /dev/null; then
  echo "REFUSING: v2_min.py still running. Finish/backup v2 first."
  exit 2
fi

echo "=== [1/4] v3 blocking rebuild (top-100) ==="
python3 v3_block_full.py > "$LOGDIR/v3_block_full.log" 2>&1 || { echo "BLOCK FAILED"; exit 1; }
ls -la tmp/test_*_cand100.npz

echo "=== [2/4] v3 training (full 150k/50k) ==="
python3 v3_train.py > "$LOGDIR/v3_train_full.log" 2>&1 || { echo "TRAIN FAILED"; exit 1; }
cat artifacts_v3/report.txt

echo "=== [3/4] v3 test scoring (two-pass) ==="
python3 v3_test.py > "$LOGDIR/v3_test.log" 2>&1 || { echo "TEST FAILED"; exit 1; }

echo "=== [4/4] validation ==="
python3 ../student_resource/student_resource/utils/validate_submission.py \
  --matching "$HOME/workspace/amazon_ml/output_v3/matching_results.tsv" \
  --candidate "$HOME/workspace/amazon_ml/output_v3/candidate_pairs.tsv" \
  > "$LOGDIR/v3_validate.log" 2>&1 || { echo "VALIDATE FAILED"; tail -20 "$LOGDIR/v3_validate.log"; exit 1; }
tail -5 "$LOGDIR/v3_validate.log"
wc -l "$HOME/workspace/amazon_ml/output_v3/"*.tsv

echo "=== copying upload candidate ==="
cp "$HOME/workspace/amazon_ml/output_v3/matching_results.tsv" "$HOME/workspace/your_files/matching_results_v3.tsv"
echo "DONE. Upload copy: ~/workspace/your_files/matching_results_v3.tsv"
