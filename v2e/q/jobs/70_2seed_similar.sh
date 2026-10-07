#!/bin/bash
# Two disclosed pairs instead of one, chosen to be as ALIKE / as UNLIKE
# each other as possible (similar).
#
# A Case-3 block pair has >=2 separated gold pairs by definition, so this
# changes nothing about which block pairs qualify. Both disclosed pairs
# are excluded from credit, which shrinks the scoring denominator, so
# compare these arms to the 1-seed baseline on pairs_gained and d|C|
# rather than on score.
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
  --n-seeds 2 --seed-pick similar \
  --order random --run-dir "$RUN" --counterfactual \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/best.txt" --tag 2seed_similar
