#!/bin/bash
# Serve script for the LongCodeZip compression service (arXiv:2510.00446).
# Used as a query-aware, two-stage code-compression baseline that compresses tool outputs.
#
# LongCodeZip needs a causal code LM to compute conditional perplexity (it cannot be served
# by vLLM), so this runs the official `longcodezip.LongCodeZip` class behind a small FastAPI
# service. Because the `longcodezip` package pins a transformers version that conflicts with
# the transformers>=5 used by .venv-vllm, install it into a dedicated venv first (uv):
#
#   uv venv .venv-longcodezip
#   uv pip install --python .venv-longcodezip -r scripts/vllm/requirements-longcodezip.txt
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME=/ssd/chr/vllm_cache/huggingface
export TORCH_HOME=/ssd/chr/vllm_cache/torch
export TMPDIR=/ssd/chr/cache/tmp
export TEMP=/ssd/chr/cache/tmp
export TMP=/ssd/chr/cache/tmp

PYTHON_BIN="${LONGCODEZIP_PYTHON:-.venv-longcodezip/bin/python}"

# Compressor LM: the paper's main setup mirrors the generation model with a ~7-8B code LM.
# Swap to Qwen/Qwen2.5-Coder-0.5B (lightweight ablation) via LONGCODEZIP_MODEL if GPUs are tight.
MODEL_NAME="${LONGCODEZIP_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

# device_map: default cuda; set LONGCODEZIP_DEVICE=cpu when GPUs are full / torch-CUDA mismatched.
DEVICE_MAP="${LONGCODEZIP_DEVICE:-cuda}"

"${PYTHON_BIN}" scripts/vllm/_serve_longcodezip.py \
  --model-name "${MODEL_NAME}" \
  --device-map "${DEVICE_MAP}" \
  --port 8005
