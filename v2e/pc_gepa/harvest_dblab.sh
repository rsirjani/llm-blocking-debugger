#!/bin/bash
# Harvest, running natively on dblab.
#
# Same experiment as pc_gepa/harvest_v2e.sh, with the laptop removed
# from the critical path. The laptop only ever orchestrated: the GPU,
# the model and now the cell bank all live here, so running here deletes
# the ssh tunnels, fail2ban, the ssh-agent and the laptop's thermal
# shutdowns from the set of things that can kill a 40 h job.
#
# The endpoint is listed three times on purpose. SLOTS_PER_SERVER (8) is
# multiplied by the number of endpoints to size the worker pool, and the
# laptop's three tunnels all pointed at this one server anyway, so this
# reproduces the tested 24-worker arrangement exactly, minus ssh.
# harness._inflight keys by URL, so the duplicates share one in-flight
# counter, which is correct for a single server.
set -euo pipefail
cd "$(dirname "$0")"

source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570

RUN=../runs/gepa/v2e_qwen38
VARIANTS=${VARIANTS:-3}

# Must match the search's flags exactly; the cell set is derived from
# them, so one differing flag silently evaluates a different matrix.
COMMON=(--blockers lambdafold simcse reclin2 klsh
        --train-datasets amazon-google walmart-amazon dblp-acm cora
        --test-datasets abt-buy dblp-scholar
        --model qwen3.8:27b-q4_K_M
        --pairs-per-instance 64 --val-per-cell 4 --test-per-cell 8
        --order random --run-dir "$RUN")

for ARM in seed best; do
  TPL="$RUN/prompts/$ARM.txt"
  [ -s "$TPL" ] || { echo "missing $TPL" >&2; exit 1; }
  echo "=== arm=$ARM variants=$VARIANTS ($(wc -c <"$TPL") chars) $(date) ==="
  python -u v2.py "${COMMON[@]}" \
      --eval-split val --variants "$VARIANTS" \
      --template-file "$TPL" --tag "$ARM" \
      2>&1 | tee "$RUN/harvest_${ARM}.log"
done

echo "=== cross-dataset test: abt-buy, dblp-scholar $(date) ==="
python -u v2.py "${COMMON[@]}" \
    --eval-split test --variants 1 \
    --template-file "$RUN/prompts/best.txt" --tag best \
    2>&1 | tee "$RUN/harvest_test_best.log"

python -u harvest_report.py "$RUN" | tee "$RUN/HARVEST_REPORT.txt"
echo "=== harvest complete $(date) ==="
