#!/bin/bash
# EXHAUSTIVE SWEEP: every Case-3 block pair in every training cell, one
# instance per cell, both arms.
#
# --pairs-per-instance 700 exceeds the largest pool (651), so
# make_instance takes each cell's pool WHOLE rather than sampling: 2,300
# block pairs across the 10 cells, visited once each in a random order.
#
# The question this answers and the 64-pair runs cannot: edits reshuffle
# the partition, so each analysed pair changes what the later pairs look
# like. Some are invalidated (blocks merged, or emptied -- now counted as
# block_dissolved), some become easier, some harder. Only a full sweep
# shows whether there is a point past which one more pair hurts.
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
  --pairs-per-instance 700 --val-per-cell 1 --test-per-cell 8 \
  --order random --run-dir "$RUN" --counterfactual \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/best.txt" --tag exh_best
