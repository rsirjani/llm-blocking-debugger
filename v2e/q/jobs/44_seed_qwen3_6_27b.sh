#!/bin/bash
# E4: judge panel. Prompt fixed, judge varied.
#
# BOTH arms are run per model. Running only the optimized prompt across
# models would be a leaderboard and would confound "this model is good
# at the task" with "this model benefits from the optimization"; the
# seed arm separates them. The instruction was optimized against
# qwen3.8:27b, so a drop elsewhere is a transfer result, not a
# capability ranking, and must be reported as such.
set -euo pipefail
cd /data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/pc_gepa
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
# ONE endpoint, not three. Worker count is SLOTS_PER_SERVER x number of
# endpoints, so this runs 8 concurrent calls rather than 24. Ollama
# pre-allocates KV for all 24 server slots regardless, but the panel
# models differ enormously in what that costs -- mistral-small3.2:24b
# holds 65.2 GB at ctx 18432 against qwen3.8:27b's 18.9 GB -- and the
# ~7 GB left over cannot feed 24 concurrent forward passes on long
# prompts. That exhaustion is what timed this job out at 92% twice.
export OLLAMA_URL=http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38

# Free VRAM before loading a different judge. Ollama sizes its KV cache
# for all 24 parallel slots, and these models differ hugely in what that
# costs: qwen3.8:27b holds 18.9 GB at ctx 18432 while
# mistral-small3.2:24b holds 65.2 GB. Two resident at once oversubscribe
# the 72 GB across the three cards, spill to host memory, and calls then
# exceed the 600 s request timeout -- which is how this job failed three
# times. Unload everything first, then load exactly one.
for M in $(curl -s -m 10 http://127.0.0.1:11570/api/ps \
           | python3 -c "import json,sys; print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))"); do
  curl -s -m 30 http://127.0.0.1:11570/api/generate \
       -d "{\"model\":\"$M\",\"keep_alive\":0}" >/dev/null || true
done
sleep 10
python -u v2.py \
  --blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets abt-buy dblp-scholar \
  --model qwen3.6:27b \
  --pairs-per-instance 32 --val-per-cell 4 --test-per-cell 8 \
  --order random --run-dir "$RUN" --counterfactual \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/seed.txt" --tag panel_qwen3_6_27b_seed
