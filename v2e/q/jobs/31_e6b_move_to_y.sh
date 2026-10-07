#!/bin/bash
# E6-B: the operation is assigned, not chosen. Same 40 instances, k=32.
set -euo pipefail
cd /data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/pc_gepa
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38
python -u v2.py \
  --blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets abt-buy dblp-scholar \
  --model qwen3.8:27b-q4_K_M \
  --pairs-per-instance 32 --val-per-cell 4 --test-per-cell 8 \
  --order random --run-dir "$RUN" --counterfactual --force-op move_to_y \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/force_move_to_y.txt" --tag force_move_to_y
