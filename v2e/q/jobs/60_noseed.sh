#!/bin/bash
# How much does the disclosed pair buy? Same blocks, same scoring (the
# instance still has a seed internally, for the already-solved check and
# for excluding it from credit) -- the judge simply is not shown it. The
# instruction had to drop its five references to the seed, so the seed's
# value is confounded with that rewrite; stated rather than hidden.
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
  --order random --run-dir "$RUN" --counterfactual  \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/noseed.txt" --tag noseed
