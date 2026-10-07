#!/bin/bash
# Health heartbeat. Appends one line every 10 minutes so a stall is
# visible after the fact, not only while someone is watching.
#
# "Stalled" is judged on completed judge calls, not on process liveness:
# the harness deliberately blocks and holds a rollout open when an
# endpoint refuses, so a wedged run looks perfectly alive in ps.
set -uo pipefail
RUN=/data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/runs/gepa/v2e_qwen38
H="$RUN/HEALTH.log"
prev=$(wc -l < "$RUN/llm_events.jsonl" 2>/dev/null || echo 0)
# Write one line at startup. A monitor whose first output is ten minutes
# away cannot be distinguished from one that failed to start, which is
# exactly what happened on the first attempt.
first=1
baseline=1
while true; do
    if [ "$first" -eq 1 ]; then first=0; else sleep 600; fi
    now=$(wc -l < "$RUN/llm_events.jsonl" 2>/dev/null || echo 0)
    rate=$(( (now - prev) * 6 ))
    procs=$(pgrep -c -f "v2\.py --blockers" 2>/dev/null); procs=${procs:-0}
    [ -z "$procs" ] && procs=0
    gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '/' )
    ol=$(curl -s -m 5 http://127.0.0.1:11570/api/version 2>/dev/null | head -c 40)
    [ -z "$ol" ] && ol="OLLAMA-DOWN"
    flag=""
    # The startup line has no interval behind it, so its rate is not a
    # measurement and must not raise a stall.
    if [ "$baseline" -eq 1 ]; then
        baseline=0; rate=-1
    elif [ "$procs" -gt 0 ] && [ "$rate" -eq 0 ]; then
        flag="  <-- STALLED: process alive, zero calls in 10 min"
    fi
    [ "$procs" -gt 1 ] && flag="$flag  <-- WARNING: $procs concurrent v2.py (serial rule violated)"
    if [ "$rate" -lt 0 ]; then rtxt="baseline"; else rtxt="$rate/h"; fi
    printf '%s procs=%s calls=%s rate=%s gpu=%s ollama=%s%s\n' \
        "$(date '+%F %T')" "$procs" "$now" "$rtxt" "$gpu" "$ol" "$flag" >> "$H"
    prev=$now
done
