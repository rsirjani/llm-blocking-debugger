#!/bin/bash
# E5 phase 1: rebuild cells at a different blocker operating point.
# Scoped to two blockers on one clean and one dirty dataset, so the
# contrast is a blocker's own tuning on fixed data rather than a change
# of dataset. Written to its own bank root; the campaign bank is not
# touched.
set -euo pipefail
cd /data/project/dblab/rsirjani/entity-matching-llm-blocking/v2e/pc_gepa
source /home/rsirjani/entity-matching-llm-blocking/env.sh
export PYTHONHASHSEED=0
export DATASETS_DIR=/home/rsirjani/entity-matching-llm-blocking/datasets
python -u build_bank2.py \
  --blockers simcse reclin2 \
  --datasets dblp-acm cora \
  --target-pc 0.45 \
  --outroot ../runs/bank_pc45

# A build that produces no cells must fail. It previously exited 0 after
# every cell errored, the queue marked it done, and the next job ran an
# evaluation against an empty bank -- "no training cells found" -- which
# reported the failure one step away from its cause.
built=$(ls -1 ../runs/bank_pc45/*/*.json 2>/dev/null | wc -l)
echo "cells built: $built"
if [ "$built" -eq 0 ]; then
  echo "build produced no cells; failing so the queue stops here" >&2
  exit 1
fi
