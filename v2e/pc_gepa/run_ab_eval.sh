#!/bin/bash
# Like-for-like: seed prompt vs best prompt on the SAME frozen validation
# instances. Instance ids are deterministic, so both arms see identical
# block pairs, seeds, samples and order -- a paired comparison.
cd "$(dirname "$0")"
source ../.venv/bin/activate
export HF_HOME=$HOME/.cache/huggingface PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11441,http://127.0.0.1:11442,http://127.0.0.1:11443
COMMON="--blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets abt-buy dblp-scholar \
  --model qwen3.8:27b-q4_K_M --pairs-per-instance 64 \
  --eval-split val --val-per-cell 4 --variants 1 --order random \
  --run-dir ../runs/gepa/v2e_qwen38"
echo "=== ARM A: SEED prompt  $(date '+%F %T')"
python -u v2.py $COMMON --template-file seed_prompt.txt --tag seedprompt
echo "=== ARM B: BEST prompt  $(date '+%F %T')"
python -u v2.py $COMMON --template-file best_prompt.txt --tag bestprompt
echo "=== DONE $(date '+%F %T')"
