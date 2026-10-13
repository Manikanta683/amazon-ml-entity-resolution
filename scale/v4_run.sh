#!/bin/bash
# v4 full overnight run — one command. Run AFTER v3_block_full.py exits
# (it waits if still alive) and after logs/v4/VALIDATION_sample.txt says PASS.
#
#   bash v4_run.sh [--pos-k 5.0] [--n-train 60000] [--n-tune 12000]
#
# Sequence: augment-test blocking -> full train -> test scoring -> validate.
set -u
cd "$(dirname "$0")"
LOGDIR=logs/v4
mkdir -p "$LOGDIR" artifacts_v4
RUNLOG="$LOGDIR/v4_run.log"
POS_K=5.0; N_TRAIN=60000; N_TUNE=12000
while [ $# -gt 0 ]; do case "$1" in
  --pos-k) POS_K="$2"; shift 2;;
  --n-train) N_TRAIN="$2"; shift 2;;
  --n-tune) N_TUNE="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;;
esac; done

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }
die() { say "FATAL: $*"; exit 1; }

say "=== v4 full run starting (pos_k=$POS_K n_train=$N_TRAIN n_tune=$N_TUNE) ==="

# 0. wait for v3_block_full.py (frees ~1.5GB)
while pgrep -f "[v]3_block_full[.]py" > /dev/null; do
  say "waiting for v3_block_full.py ..."; sleep 120
done
say "v3_block_full.py exited"

# 0b. crash recovery: france npz may be missing (np.bytes_.encode bug in the
# first run) -> rebuild just that shard with the fixed script
if [ ! -f tmp/test_france_cand100.npz ]; then
  say "test_france_cand100.npz MISSING -> rebuilding france shard"
  python3 v3_block_full.py --shards france >> "$RUNLOG" 2>&1 \
    || die "france rebuild failed"
fi
for s in france india us; do
  [ -f "tmp/test_${s}_cand100.npz" ] || die "tmp/test_${s}_cand100.npz missing"
done
say "all test cand100 npz present"

# 1. augmented test blocking (+ vec copy into artifacts_v4)
[ -f artifacts_v4/fold_dict.pkl ] || die "artifacts_v4/fold_dict.pkl missing (run v4_mine.py)"
say "-- augmenting test blocking --"
python3 v4_block_aug.py --augment-test >> "$RUNLOG" 2>&1 \
  || die "augment-test failed"
say "augment-test done"

# 2. full train (builds v4 train blocking per shard, trains HGB)
say "-- full training (n_train=$N_TRAIN) --"
python3 v4_train.py --full --n-train "$N_TRAIN" --n-tune "$N_TUNE" \
  --pos-k "$POS_K" >> "$RUNLOG" 2>&1 \
  || die "v4_train --full failed"
say "training done: $(cat artifacts_v4/report.txt)"

# 3. test scoring
say "-- test scoring --"
python3 v4_test.py >> "$RUNLOG" 2>&1 || die "v4_test failed"
say "scoring done"

# 4. validate outputs
say "-- validating outputs --"
python3 - <<'EOF' | tee -a "$RUNLOG"
import os, sys
sys.path.insert(0, os.getcwd())
from common import NORM, read_chunks
out = os.path.expanduser("~/workspace/amazon_ml/output_v4")
n_exp = sum(sum(1 for _ in ch) for ch in
            read_chunks(os.path.join(NORM, "test_source1.tsv"),
                        usecols=["entity_id"]))
for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
    p = os.path.join(out, fn)
    assert os.path.exists(p), f"missing {fn}"
    with open(p) as f:
        hdr = f.readline().strip()
        n = sum(1 for _ in f)
    print(f"{fn}: header=[{hdr}] rows={n:,}")
    assert n == n_exp, f"row count {n} != test S1 {n_exp}"
# matched subset of candidates (spot check first 200k rows)
mr, cp = {}, {}
with open(os.path.join(out, "matching_results.tsv")) as f:
    f.readline()
    for i, line in enumerate(f):
        if i >= 200000: break
        sid, _, m = line.rstrip("\n").partition("\t")
        mr[sid] = set(m.split(",")) if m else set()
with open(os.path.join(out, "candidate_pairs.tsv")) as f:
    f.readline()
    for i, line in enumerate(f):
        if i >= 200000: break
        sid, _, c = line.rstrip("\n").partition("\t")
        cp[sid] = set(c.split(",")) if c else set()
bad = sum(1 for sid in mr if not mr[sid] <= cp[sid])
print(f"matched-subset-of-candidates violations (200k spot): {bad}")
assert bad == 0
import numpy as np
cs = []
with open(os.path.join(out, "candidate_pairs.tsv")) as f:
    f.readline()
    for i, line in enumerate(f):
        if i >= 200000: break
        cs.append(line.count(",") + (1 if line.rstrip("\n").partition("\t")[2] else 0))
print(f"mean candidates/S1 (200k spot): {np.mean(cs):.1f} max={max(cs)}")
print("VALIDATION OK")
EOF
say "=== v4 full run COMPLETE ==="
say "upload file: ~/workspace/amazon_ml/output_v4/matching_results.tsv"
