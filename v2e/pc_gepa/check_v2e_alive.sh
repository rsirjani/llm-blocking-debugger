#!/bin/bash
# Liveness check for the v2e campaign, meant for cron.
# Watches the ONE thing that matters -- has a judge call landed recently
# -- rather than whether some supervisor process still exists. Two
# supervisor processes have now died silently and taken the run with
# them; the event log cannot lie the same way.
RUN=/home/tin/projects/entity-matching-llm-blocking/runs/gepa/v2e_qwen38
DIR=/home/tin/projects/entity-matching-llm-blocking/pc_gepa
STALE_MIN=${1:-30}

age=$(python3 -c "
import json,time
last=0
for line in open('$RUN/llm_events.jsonl'):
    try:
        e=json.loads(line)
    except Exception:
        continue
    if e.get('tag')!='reflection-sonnet':
        last=max(last,e['ts'])
print(int((time.time()-last)/60))
" 2>/dev/null) || exit 0
[ -z "$age" ] && exit 0
[ "$age" -lt "$STALE_MIN" ] && exit 0

echo "$(date '+%F %T') no judge call for ${age} min -- inspecting"

# Repair tunnels only if they are genuinely unreachable.
for p in 11441 11442 11443; do
  curl -s --max-time 8 "http://127.0.0.1:$p/api/version" >/dev/null && continue
  for pid in $(ss -tlnp 2>/dev/null | grep ":$p " | grep -o 'pid=[0-9]*' \
               | cut -d= -f2 | sort -u); do kill "$pid" 2>/dev/null; done
  sleep 2
  setsid ssh -o ControlPath=none -o ControlMaster=no -o ConnectTimeout=20 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes -N -L "$p:127.0.0.1:11570" dblab-gpu \
    >/dev/null 2>&1 &
  echo "  reopened tunnel $p"
done

# Relaunch ONLY if the driver process is gone. A live driver with stale
# judge calls is most likely waiting on a reflection retry, which now
# backs off for up to 30 minutes at a time against subscription usage
# limits. Killing it there would destroy the very step we are waiting
# for and restart the loop that lost two days.
if pgrep -f "v2[.]py --blockers" >/dev/null; then
  echo "  driver alive -- not touching it (likely awaiting reflection)"
  exit 0
fi
sleep 10
# Relaunch as a transient user unit, not a setsid child: a child forked
# here still lives in this oneshot's cgroup and is killed the moment the
# service deactivates (the likely mechanism behind the silently-dying
# supervisors). MemoryHigh throttles a leaky run before it can drag the
# whole machine into swap-thrash (the Sep 2026 freeze root cause).
systemd-run --user --collect --unit v2e-campaign \
  -p MemoryHigh=8G -p WorkingDirectory="$DIR" \
  bash -c "./launch_v2e.sh >> '$RUN/driver.log' 2>&1"
echo "  driver was dead -- relaunched (transient unit v2e-campaign, MemoryHigh=8G)"
