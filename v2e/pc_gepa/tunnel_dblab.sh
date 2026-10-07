#!/bin/bash
# Holds three ssh tunnels, all to the ONE Ollama server on dblab:
#   local 11441/11442/11443 -> dblab 11570
# Three separate GPU-pinned servers did not work: Ollama 0.33.2 ignores
# CUDA_VISIBLE_DEVICES (by index or by UUID) and layer-split every model
# across all three cards, so we paid 3x weights, 3x KV and 3x
# coordination for the same hardware -- 12 of 16 cores and 4x latency.
# One server with 12 parallel slots holds a single copy. The driver
# still dispatches over three local ports, which costs nothing and means
# its OLLAMA_URL never has to change.
#
# Deliberately gentle: checks every 5 minutes, requires two consecutive
# failures before acting, one connection attempt per cycle per port. An
# earlier version retried every 60s with no usable key and produced
# enough failed authentications to get this host banned by fail2ban.
# Never tighten this loop.
declare -A MAP=( [11441]=11570 [11442]=11570 [11443]=11570 )
declare -A FAILS=( [11441]=0 [11442]=0 [11443]=0 )

# ssh-agent.service is socket-activated, so the agent process can be
# replaced while this script runs, and the replacement holds no keys --
# the passphrase-protected key can only be re-added by hand. Every
# connection attempt made in that window is a guaranteed
# "Permission denied (publickey)" on the far end, and five of those
# inside ten minutes is the fail2ban threshold. So: no key, no attempt.
# This does not shorten the interval or add retries; it only declines to
# spend an attempt that cannot possibly succeed and can only cost us a
# ban. ssh-add -l exits 0 only when the agent actually holds a key.
NO_KEY_LOGGED=0
agent_has_key() {
  ssh-add -l >/dev/null 2>&1
}

open_tunnel() {
  local lport=$1 rport=$2
  if ! agent_has_key; then
    if [ "$NO_KEY_LOGGED" -eq 0 ]; then
      echo "$(date '+%F %T') ssh-agent holds no key; not attempting" \
           "${lport}. Run: ssh-add ~/.ssh/id_ed25519_dblab"
      NO_KEY_LOGGED=1
    fi
    return 1
  fi
  if [ "$NO_KEY_LOGGED" -eq 1 ]; then
    echo "$(date '+%F %T') key is back in the agent; resuming tunnels"
    NO_KEY_LOGGED=0
  fi
  for pid in $(ss -tlnp 2>/dev/null | grep ":${lport} " \
               | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
    kill "$pid" 2>/dev/null
  done
  sleep 2
  setsid ssh -o ControlPath=none -o ControlMaster=no \
    -o ConnectTimeout=20 -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -N -L "${lport}:127.0.0.1:${rport}" dblab-gpu >/dev/null 2>&1 &
  echo "$(date '+%F %T') opened ${lport} -> dblab:${rport}"
}

healthy() {
  curl -s --max-time 10 "http://127.0.0.1:$1/api/version" >/dev/null 2>&1
}

# Only open a port that is not already serving. Startup used to kill and
# rebuild unconditionally, which meant adopting a live set of tunnels --
# or a systemd restart -- dropped the driver's endpoint for a few
# seconds for no reason. Under a unit with Restart=always that happened
# on every restart.
for lport in "${!MAP[@]}"; do
  if healthy "$lport"; then
    echo "$(date '+%F %T') ${lport} already serving; adopting"
  else
    open_tunnel "$lport" "${MAP[$lport]}"
  fi
done
sleep 10

while true; do
  for lport in "${!MAP[@]}"; do
    if curl -s --max-time 10 "http://127.0.0.1:${lport}/api/version" \
         >/dev/null 2>&1; then
      FAILS[$lport]=0
    else
      FAILS[$lport]=$(( FAILS[$lport] + 1 ))
      if [ "${FAILS[$lport]}" -ge 2 ]; then
        open_tunnel "$lport" "${MAP[$lport]}"
        FAILS[$lport]=0
        sleep 30
      fi
    fi
  done
  sleep 300
done
