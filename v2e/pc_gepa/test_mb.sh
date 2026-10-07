#!/bin/bash
# Cross-dataset arm extended to MusicBrainz (dirty ER, out of the
# training matrix).
#
# Run with --test-datasets musicbrainz-20k ALONE rather than appended to
# the original two. Instance ids are keyed by blocker and dataset
# (test:<blocker>:<dataset>:v0:<j>), not by position in the cell list,
# so the ids produced here are identical to those a combined run would
# produce and the results merge with eval_test_best.json. Running it
# alone avoids re-spending 2,048 calls reproducing abt-buy and
# dblp-scholar, which under a pinned server would come back bit-exact.
set -euo pipefail
cd "$(dirname "$0")"
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11570,http://127.0.0.1:11570,http://127.0.0.1:11570
RUN=../runs/gepa/v2e_qwen38
python -u v2.py \
  --blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets musicbrainz-20k \
  --model qwen3.8:27b-q4_K_M \
  --pairs-per-instance 64 --val-per-cell 4 --test-per-cell 8 \
  --order random --run-dir "$RUN" \
  --eval-split test --variants 1 \
  --template-file "$RUN/prompts/best.txt" --tag mbbest \
  2>&1 | tee "$RUN/harvest_mb_best.log"
echo "=== musicbrainz best arm complete $(date) ==="
