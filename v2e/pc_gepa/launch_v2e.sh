#!/bin/bash
# Lives in the project, not /tmp: a reboot wiped /tmp on 2026-09-08 and
# took the launch script and run log with it, so the campaign could not
# be restarted without rebuilding them from scratch.
cd "$(dirname "$0")"
source ../.venv/bin/activate
export HF_HOME=$HOME/.cache/huggingface PYTHONHASHSEED=0
# Three tunnels, all to the ONE Ollama server on dblab (11570). Ollama
# 0.33.2 ignores CUDA_VISIBLE_DEVICES and layer-splits across all three
# cards regardless, so three "pinned" servers meant three model copies
# fighting over the same GPUs. One copy, 24 slots, dispatched over three
# local ports by the client's least-loaded picker.
export OLLAMA_URL=http://127.0.0.1:11441,http://127.0.0.1:11442,http://127.0.0.1:11443
# qwen3.8:27b judge, thinking off (thinking is compute-bound at 7.4s/call
# regardless of concurrency and would take weeks).
# minibatch 24, not 8: at 8 the accept gate let through 2 of 3 candidates
# that were WORSE than their parent on the 40-instance val panel.
# val-per-cell 4 = 40 validation instances. Budget counts INSTANCES, so
# panel size costs no wall-clock -- it moves budget from exploration to
# selection.
# Cells are admitted whole: any cell where >10% of its Case-3 block pairs
# cannot be shown in full is dropped, so the sample is never trimmed.
# GEPA resumes from gepa_state.bin in the run dir, so re-running this
# after a crash or reboot picks up where it left off.
exec python -u v2.py \
  --blockers lambdafold simcse reclin2 klsh \
  --train-datasets amazon-google walmart-amazon dblp-acm cora \
  --test-datasets abt-buy dblp-scholar \
  --model qwen3.8:27b-q4_K_M \
  --pairs-per-instance 64 --budget 5000 --val-per-cell 4 \
  --test-per-cell 8 --minibatch 24 \
  --order random --run-dir ../runs/gepa/v2e_qwen38
