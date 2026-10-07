#!/bin/bash
# EXHAUSTIVE SWEEP REPLICATE, VARIANT v1, optimized-prompt arm.
# Identical to 81_exhaustive_best.sh except: instances are v1 (different
# visit order + disclosed seeds over the SAME boundary pools) and the
# output is eval_val_exh_best_v1.json. Tests whether the v0 headline
# survives a different ordering (FACTS_CORE 0.6, VERIFY_R1 section 7).
set -euo pipefail
# The queue's wait_idle only matches "v2.py --blockers"; the serving
# spot-check runs as /tmp/spotcheck_driver.py and shares the single
# llama-server slot, so wait for it here too.
while pgrep -f "v2\.py --blockers|spotcheck_driver\.py" >/dev/null 2>&1; do
    echo "$(date '+%F %T') waiting: judge work still running"
    sleep 60
done
cd /data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/pc_gepa
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38
# Originals are irreplaceable; the replicate must not clobber anything.
for F in eval_val_exh_seed.json eval_val_exh_best.json; do
    [ -s "$RUN/$F" ] || { echo "sanity: $RUN/$F missing"; exit 1; }
done
[ -e "$RUN/eval_val_exh_best_v1.json" ] && { echo "output exists; refusing to overwrite"; exit 1; }
curl -s -m 8 http://127.0.0.1:11570/api/ps  > "$RUN/serving_exh_v1_best_before.json" || true
curl -s -m 5 http://127.0.0.1:11570/api/version >> "$RUN/serving_exh_v1_best_before.json" || true
python -u exh_v1_from_v2.py \
  --blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets abt-buy dblp-scholar \
  --model qwen3.8:27b-q4_K_M \
  --pairs-per-instance 700 --val-per-cell 1 --test-per-cell 8 \
  --order random --run-dir "$RUN" --counterfactual \
  --eval-split val --variants 1 \
  --template-file "$RUN/prompts/best.txt" --tag exh_best_v1
curl -s -m 8 http://127.0.0.1:11570/api/ps  > "$RUN/serving_exh_v1_best_after.json" || true
curl -s -m 5 http://127.0.0.1:11570/api/version >> "$RUN/serving_exh_v1_best_after.json" || true
