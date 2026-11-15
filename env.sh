#!/usr/bin/env bash
# Activate your Python environment before sourcing this file.
export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ICLR${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRITON_CACHE_DIR="${PROJECT_ROOT}/.cache/triton"
export ALPACA_EVAL_ASSETS="${PROJECT_ROOT}/alpaca_assets"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
