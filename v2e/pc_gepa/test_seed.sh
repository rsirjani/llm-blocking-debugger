#!/bin/bash
# Seed baseline on the cross-dataset cells.
#
# HELDOUT_SPEC section 2 ran only the optimized arm here, which shows
# that candidate 7 works on datasets never trained on but gives nothing
# to compare it against. Without this arm the cross-dataset result is an
# absolute number, not a transfer claim about the optimization. Same
# frozen specs, same flags, same server, so it pairs instance-for-
# instance with eval_test_best.json.
set -euo pipefail
cd "$(dirname "$0")"
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38
python -u v2.py \
  --blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets abt-buy dblp-scholar \
  --model qwen3.8:27b-q4_K_M \
  --pairs-per-instance 64 --val-per-cell 4 --test-per-cell 8 \
  --order random --run-dir "$RUN" \
  --eval-split test --variants 1 \
  --template-file "$RUN/prompts/seed.txt" --tag seed \
  2>&1 | tee "$RUN/harvest_test_seed.log"
echo "=== test seed arm complete $(date) ==="
