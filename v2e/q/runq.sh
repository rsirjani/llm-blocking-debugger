#!/bin/bash
# Serial experiment runner.
#
# Local-judge work must not overlap: two processes share the 24 Ollama
# slots and the same llm_events.jsonl, whose prompt payloads exceed the
# 4 KB atomic-append window, so concurrent writers can tear lines in the
# file the paper's numbers come out of. This runner therefore never
# starts a job while any v2.py is alive -- including one launched by
# hand outside the queue, which is how E11 was started.
#
# Jobs are jobs/NN_name.sh, run in lexical order. A job is considered
# done when its marker exists, so the queue is resumable: re-running the
# script after a crash picks up where it stopped rather than repeating
# completed arms.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# Exactly one runner, enforced by the kernel rather than by hoping the
# chainer never fires twice. Two runners would each pass wait_idle the
# moment a job ended and could then start the same job, or different
# jobs concurrently, which is the serial rule broken in the one place it
# matters.
exec 9>"$ROOT/.runq.lock"
flock -n 9 || { echo "$(date '+%F %T') another runner holds the lock; exiting" \
                >> "$ROOT/../runs/gepa/v2e_qwen38/QUEUE.log"; exit 0; }
RUN=/data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/runs/gepa/v2e_qwen38
LOG="$RUN/QUEUE.log"
STATUS="$RUN/QUEUE_STATUS.txt"
mkdir -p "$ROOT/markers"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

wait_idle() {
    local waited=0
    while pgrep -f "v2\.py --blockers" >/dev/null 2>&1; do
        [ $((waited % 600)) -eq 0 ] && \
            say "waiting: another v2.py is running (${waited}s)"
        # 9>&- so the child does not inherit the lock fd: a runner
        # killed mid-sleep would otherwise leave an orphan holding the
        # lock until it woke.
        sleep 60 9>&-; waited=$((waited+60))
    done
}

# Serving configuration is a controlled variable (HELDOUT_SPEC section
# 7): arms separated by a restart or model reload are not comparable, so
# record it with every job rather than trusting it did not change.
pin() {
    local v d
    v=$(curl -s -m 5 http://127.0.0.1:11570/api/version)
    d=$(curl -s -m 8 http://127.0.0.1:11570/api/ps | head -c 400)
    say "server pin: version=$v"
    say "server pin: ps=$d"
}

say "queue starting, $(ls "$ROOT"/jobs/*.sh 2>/dev/null | wc -l) jobs"
pin

for J in "$ROOT"/jobs/*.sh; do
    [ -e "$J" ] || continue
    NAME=$(basename "$J" .sh)
    MARK="$ROOT/markers/$NAME.done"
    if [ -f "$MARK" ]; then say "skip $NAME (done $(cat "$MARK"))"; continue; fi
    wait_idle
    say "START $NAME"
    printf 'running %s since %s\n' "$NAME" "$(date '+%F %T')" > "$STATUS"
    # 9>&- matters most here. A job runs for hours; if the runner were
    # killed while it ran, an inherited lock fd would keep the lock held
    # for the rest of the job and the chainer could never relaunch.
    bash "$J" 9>&- >> "$RUN/${NAME}.out" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
        date '+%F %T' > "$MARK"; say "DONE  $NAME"
    else
        say "FAIL  $NAME rc=$RC -- stopping the queue rather than running"
        say "      the next arm against an unknown state"
        # Sentinel so the chainer does not undo the fail-stop. Without
        # it the chainer sees pending work, relaunches, and the same job
        # fails again: three cycles of that burned ~21 h on 41_best.
        printf '%s failed at %s rc=%s\n' "$NAME" "$(date '+%F %T')" "$RC" \
            > "$ROOT/QUEUE_FAILED"
        printf 'FAILED %s rc=%s at %s\n' "$NAME" "$RC" "$(date '+%F %T')" > "$STATUS"
        exit $RC
    fi
done
say "queue complete"
printf 'idle -- all jobs complete %s\n' "$(date '+%F %T')" > "$STATUS"
