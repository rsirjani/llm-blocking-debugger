#!/bin/bash
# Hold the harvest until iteration 127 is banked, then hand off.
#
# 127 already ran to a verdict on Sat 19 Sep (rejected on the paired
# minibatch gate: subsample 10.815 vs 12.667) but the machine lost power
# before the state save, so gepa_state.bin still reads i=126. The driver
# is re-running it. Waiting costs a few hours and buys a clean count:
# 16 gate-tested iterations, all of them banked, with no footnote
# explaining why the log records a verdict the state does not.
#
# Banking is detected from run_log.txt rather than gepa_state.bin: GEPA
# writes state at the end of an iteration, so the first "Iteration 128:"
# line means 127 is closed and saved. Text, and no need to deserialise
# the engine's state to make a control-flow decision.
#
# Nothing else depends on this. If the wait runs long, trigger the
# harvest by hand -- the harvest does not read i.
set -uo pipefail

LOG=/home/tin/projects/entity-matching-llm-blocking/runs/gepa/v2e_qwen38/run_log.txt
NEXT="Iteration 128:"
MAX_WAIT_H=${MAX_WAIT_H:-10}

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

say() { echo "$(date '+%F %T') $*"; }

say "waiting for '$NEXT' to appear in run_log.txt (max ${MAX_WAIT_H}h)"

while :; do
    if grep -qF "$NEXT" "$LOG" 2>/dev/null; then
        say "iteration 127 closed and banked -- starting harvest"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        say "waited ${MAX_WAIT_H}h without 127 closing; starting harvest" \
            "anyway. Report 15 banked / 16 gate-tested -- the verdict for" \
            "127 is in run_log.txt near line 5301, not in the state file."
        break
    fi
    sleep 300
done

exec /usr/bin/systemctl --user start v2e-harvest.service
