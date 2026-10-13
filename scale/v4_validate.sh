#!/bin/bash
# v4 validation chain (background): waits for v3_block_full.py, recovers a
# crashed france shard, runs step7/step8 prototypes, then the v4 sample
# validation. Writes logs/v4/VALIDATION_sample.txt (+ launch command).
set -u
cd "$(dirname "$0")"
LOGDIR=logs/v4
mkdir -p "$LOGDIR" artifacts_v4
CHAINLOG="$LOGDIR/validate_chain.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$CHAINLOG"; }
die() { say "FATAL: $*"; exit 1; }

say "=== v4 validation chain started ==="

# robust wait: proceed only when ALL test npz exist AND no blocker runs
# (a dead blocker alone must NOT release the gate)
while true; do
  ok=true
  for s in france india us; do
    [ -f "tmp/test_${s}_cand100.npz" ] || ok=false
  done
  if $ok && ! pgrep -f "[v]3_block_full[.]py" > /dev/null; then
    break
  fi
  say "waiting for test blocking (npz present: $(ls tmp/test_*_cand100.npz 2>/dev/null | wc -l)/3) ..."
  sleep 120
done
say "test blocking complete"

if [ ! -f tmp/test_france_cand100.npz ]; then
  say "RECOVERY: test_france_cand100.npz missing (save-crash) -> rebuilding"
  python3 v3_block_full.py --shards france >> "$CHAINLOG" 2>&1 \
    || die "france rebuild failed"
  say "france rebuild done"
fi

say "-- step7: weighted hard-positive prototype --"
python3 logs/rd_990/step7_weighted_proto.py >> "$CHAINLOG" 2>&1 \
  || die "step7 failed"
say "-- step8: hard-mix prototype --"
python3 logs/rd_990/step8_hardneg_proto.py >> "$CHAINLOG" 2>&1 \
  || die "step8 failed"

# pick pos_k from step7's winner
POS_K=$(python3 - <<'EOF'
import re
best_k, best_d = 5.0, -1e9
txt = open("logs/rd_990/07_weighted_proto.txt").read()
for m in re.finditer(r"wpos_k=([\d.]+).*?delta vs uniform: ([+-][\d.]+)",
                     txt, re.S):
    k, d = float(m.group(1)), float(m.group(2))
    if d > best_d:
        best_k, best_d = k, d
print(best_k)
EOF
)
say "step7 winner pos_k=$POS_K"

say "-- v4 sample validation (proto 1400+600 S1) --"
python3 v4_train.py --sample --proto-dir logs/rd_990 --pos-k "$POS_K" \
  >> "$CHAINLOG" 2>&1 || die "v4 sample validation failed"

cat >> "$LOGDIR/VALIDATION_sample.txt" <<EOF

OVERNIGHT FULL-RUN LAUNCH (after PASS verdict above):
  cd ~/workspace/amazon_ml/scale && bash v4_run.sh --pos-k $POS_K --n-train 60000 --n-tune 12000
EOF
say "=== validation chain COMPLETE: logs/v4/VALIDATION_sample.txt ==="
cat "$LOGDIR/VALIDATION_sample.txt" | tee -a "$CHAINLOG" | tail -20
