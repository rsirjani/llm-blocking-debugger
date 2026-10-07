#!/bin/bash
# Run ON dblab. One server, one model copy across all three A5000s.
# Ollama 0.33.2 ignores CUDA_VISIBLE_DEVICES (index or UUID) and splits
# every model over all cards, so three "pinned" servers just meant three
# copies contending -- 57 GB of weights, 12 of 16 cores, 4x latency.
cd ~/ollama-new
export OLLAMA_HOST=127.0.0.1:11570
export OLLAMA_NUM_PARALLEL=24
export OLLAMA_KEEP_ALIVE=168h
export OLLAMA_MAX_LOADED_MODELS=1
export LD_LIBRARY_PATH=$PWD/lib/ollama
exec ./bin/ollama serve
