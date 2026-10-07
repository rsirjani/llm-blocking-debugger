#!/bin/bash
# E5 phase 2: does LLM debugging still pay when the blocker is already
# good (or already hopeless)? PC filters are deliberately widened here:
# the whole question requires leaving the [0.45, 0.80] band that admits
# every other cell in the paper, so these cells are NOT comparable to
# the main tables and are reported separately.
set -euo pipefail
cd /data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/pc_gepa
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38
python -u v2.py \
  --blockers simcse reclin2 --train-datasets dblp-acm cora \
  --test-datasets abt-buy \
  --model qwen3.8:27b-q4_K_M \
  --pairs-per-instance 32 --val-per-cell 4 --test-per-cell 8 \
  --min-pc 0.20 --max-pc 0.95 \
  --bank-root ../runs/bank_pc75 \
  --order random --run-dir "$RUN" --counterfactual \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/best.txt" --tag e5_pc75_best
