#!/bin/bash
# Relaunch the queue when it exits with work still outstanding.
#
# runq.sh expands jobs/*.sh once at startup, so jobs added while it is
# running are invisible to it. Rather than make the runner re-scan (and
# risk two runners racing for the same job), this waits for the current
# runner to exit and starts a fresh one. Done-markers make that safe:
# completed jobs are skipped, so a restart resumes rather than repeats.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec 8>"$ROOT/.chain.lock"
flock -n 8 || exit 0          # one chainer only
RUN=/data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/runs/gepa/v2e_qwen38
say() { echo "$(date '+%F %T') [chain] $*" >> "$RUN/QUEUE.log"; }
while true; do
    sleep 120
    if [ -f "$ROOT/QUEUE_FAILED" ]; then
        say "a job failed ($(cat "$ROOT/QUEUE_FAILED")); not relaunching."
        say "clear q/QUEUE_FAILED once the cause is fixed."
        exit 1
    fi
    pgrep -f "runq\.sh" >/dev/null 2>&1 && continue
    pending=0
    for J in "$ROOT"/jobs/*.sh; do
        [ -e "$J" ] || continue
        [ -f "$ROOT/markers/$(basename "$J" .sh).done" ] || pending=1
    done
    [ "$pending" -eq 0 ] && { say "no pending jobs; chain exiting"; exit 0; }
    say "runner gone with jobs pending; relaunching"
    cd "$ROOT" && setsid nohup ./runq.sh > /dev/null 2>&1 < /dev/null &
    sleep 30
done
