#!/bin/bash
# Harvest: score the seed prompt and the winning prompt on the SAME
# frozen variants of the val cells.
#
# What this measures, precisely: an instance is a whole-cell replay, so
# every variant scores the same partition, the same records and the same
# gold. Variants differ only in which Case-3 block pairs get visited,
# which failure is used as the seed in each, and in what order. So this
# is region transfer within a cell -- NOT dataset transfer and NOT cell
# transfer. Both of those would need cells that are not in train, and
# every high-degree (cora) cell is in train.
#
# v0 is the panel GEPA selected against, so it is a CONTROL, not a
# result: the best arm's v0 mean must reproduce the val mean the search
# reported. If it does not, this harness is wrong and v1+ means nothing.
# v1 upward are the instances no candidate was ever selected on.
#
# Run this only after the search has stopped. It contends for the same
# 24 Ollama slots; running it alongside the search halves both.
set -euo pipefail
cd "$(dirname "$0")"
source ../.venv/bin/activate
export HF_HOME=$HOME/.cache/huggingface PYTHONHASHSEED=0
export OLLAMA_URL=http://127.0.0.1:11441,http://127.0.0.1:11442,http://127.0.0.1:11443

RUN=../runs/gepa/v2e_qwen38
VARIANTS=${VARIANTS:-3}

# Cell construction flags must match launch_v2e.sh EXACTLY. The cell set
# is derived from these (min/max PC, oversize fraction, min Case-3), so
# a single differing flag silently evaluates a different matrix.
COMMON=(--blockers lambdafold simcse reclin2 klsh
        --train-datasets amazon-google walmart-amazon dblp-acm cora
        --test-datasets abt-buy dblp-scholar
        --model qwen3.8:27b-q4_K_M
        --pairs-per-instance 64 --val-per-cell 4 --test-per-cell 8
        --order random --run-dir "$RUN")

for ARM in seed best; do
  TPL="$RUN/prompts/$ARM.txt"
  [ -s "$TPL" ] || { echo "missing $TPL -- dump the prompts first" >&2; exit 1; }
  echo "=== arm=$ARM variants=$VARIANTS ($(wc -c <"$TPL") chars) ==="
  python -u v2.py "${COMMON[@]}" \
      --eval-split val --variants "$VARIANTS" \
      --template-file "$TPL" --tag "$ARM" \
      2>&1 | tee "$RUN/harvest_${ARM}.log"
done

# Cross-dataset transfer: abt-buy and dblp-scholar, never trained on.
# One variant only -- these cells were never selected against, so extra
# variants buy diversity we are not short of, and the slot is better
# spent on the val variants above.
#
# Both are clean 1:1, so this CANNOT exercise the dislodge failure mode
# (an edit moving records already co-blocked with their own partners).
# A good number here is a dataset-transfer claim, not a diagnosis.
# See HELDOUT_SPEC.md section 3.
echo "=== cross-dataset test: abt-buy, dblp-scholar ==="
python -u v2.py "${COMMON[@]}" \
    --eval-split test --variants 1 \
    --template-file "$RUN/prompts/best.txt" --tag best \
    2>&1 | tee "$RUN/harvest_test_best.log"

python -u harvest_report.py "$RUN" | tee "$RUN/HARVEST_REPORT.txt"
