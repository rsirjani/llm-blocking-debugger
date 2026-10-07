#!/bin/bash
# E11: leave-one-out ablation of candidate 7's seven method steps.
#
# Section VII attributes specific behaviour to specific prompt lines,
# but adjacent candidates differ by many lines at once (cand5->cand6 is
# +19/-10 at similarity 0.454), and candidates are selected on score, so
# every cross-candidate comparison is conditioned on winning. Removing
# one step at a time assigns the treatment ourselves, which is the only
# way those causal claims become testable.
#
# k=32 by decision D2: the advantage grows with k but the
# effect-to-standard-error ratio peaks near k=23-31, so 32 maximises
# power per instance at half the cost of 64.
#
# --counterfactual records what each operation would have bought with
# the predicate held fixed (E6-A), which rides along for free.
set -euo pipefail
cd "$(dirname "$0")"
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38

for N in 1 2 3 4 5 6 7; do
  TPL="$RUN/prompts/abl_no${N}.txt"
  [ -s "$TPL" ] || { echo "missing $TPL" >&2; exit 1; }
  echo "=== ablation: step $N removed  $(date) ==="
  python -u v2.py \
    --blockers lambdafold simcse reclin2 klsh \
    --train-datasets amazon-google walmart-amazon dblp-acm cora \
    --test-datasets abt-buy dblp-scholar \
    --model qwen3.8:27b-q4_K_M \
    --pairs-per-instance 32 --val-per-cell 4 --test-per-cell 8 \
    --order random --run-dir "$RUN" --counterfactual \
    --eval-split val --variants 1 \
    --template-file "$TPL" --tag "abl${N}" \
    2>&1 | tee "$RUN/e11_abl${N}.log"
done
echo "=== E11 complete $(date) ==="
