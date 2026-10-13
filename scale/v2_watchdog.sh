#!/bin/bash
# Watchdog for v2_min.py scoring run. Restarts it if dead and output incomplete (max 3 restarts).
OUT=~/workspace/amazon_ml/output_v2/matching_results.tsv
LOG=~/workspace/amazon_ml/scale/v2_watchdog.log
CTR=~/workspace/amazon_ml/scale/v2_restart_count.txt
if [ ! -f "$CTR" ]; then echo 0 > "$CTR"; fi
# complete if merged output exists and is large (final ~90MB+)
if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 10000000 ]; then
  echo "$(date '+%F %T'): output complete, watchdog standing down" >> "$LOG"
  exit 0
fi
if pgrep -f "v2_min.py" > /dev/null; then
  exit 0
fi
n=$(cat "$CTR")
if [ "$n" -ge 3 ]; then
  echo "$(date '+%F %T'): restart limit (3) reached, NOT restarting" >> "$LOG"
  exit 1
fi
echo $((n+1)) > "$CTR"
echo "$(date '+%F %T'): v2_min.py dead, restarting (attempt $((n+1))/3)" >> "$LOG"
cd ~/workspace/amazon_ml/scale && nohup python3 v2_min.py > ~/workspace/amazon_ml/scale/v2_min_watchdog.log 2>&1 &
echo "$(date '+%F %T'): restarted pid $!" >> "$LOG"
