#!/bin/bash
# Restart the driver at the next iteration boundary, so a code change
# takes effect without discarding an in-flight rollout/retest.
#
# GEPA commits total_num_evals to gepa_state.bin at the START of the
# next iteration, so a rise above the baseline means the previous
# iteration closed and its work is safely on disk. Restarting then
# costs only the handful of calls the new iteration has just started.
cd "$(dirname "$0")"
RUN=../runs/gepa/v2e_qwen38
PY=../.venv/bin/python
BASE=$("$PY" -c "
import pickle
print(pickle.load(open('$RUN/gepa_state.bin','rb'))['total_num_evals'])")
echo "$(date '+%F %T') waiting for evals to rise above $BASE"
while true; do
  CUR=$("$PY" -c "
import pickle
try: print(pickle.load(open('$RUN/gepa_state.bin','rb'))['total_num_evals'])
except Exception: print($BASE)" 2>/dev/null)
  if [ -n "$CUR" ] && [ "$CUR" -gt "$BASE" ] 2>/dev/null; then
    echo "$(date '+%F %T') iteration committed ($BASE -> $CUR); restarting"
    for pid in $(pgrep -f "v2[.]py --blockers"); do kill "$pid" 2>/dev/null; done
    sleep 8
    setsid nohup ./launch_v2e.sh >> "$RUN/driver.log" 2>&1 </dev/null &
    sleep 20
    pgrep -f "v2[.]py --blockers" >/dev/null \
      && echo "$(date '+%F %T') relaunched; staged code changes now live" \
      || echo "$(date '+%F %T') RELAUNCH FAILED -- liveness timer will retry"
    exit 0
  fi
  sleep 120
done
