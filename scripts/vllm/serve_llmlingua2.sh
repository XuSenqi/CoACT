#!/bin/bash
# Serve script for the LLMLingua-2 compression service (XLM-RoBERTa-large token classifier).
# Used as a task-agnostic baseline compression strategy that compresses tool outputs.
#
# LLMLingua-2 cannot be served by vLLM (it is an encoder model), so this runs the official
# `llmlingua.PromptCompressor` behind a small FastAPI service. Because `llmlingua` conflicts
# with the transformers>=5 pinned by .venv-vllm, install it into a dedicated venv first (uv):
#
#   uv venv .venv-llmlingua
#   uv pip install --python .venv-llmlingua -r scripts/vllm/requirements-llmlingua2.txt
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME=/ssd/chr/vllm_cache/huggingface
export TORCH_HOME=/ssd/chr/vllm_cache/torch
export TMPDIR=/ssd/chr/cache/tmp
export TEMP=/ssd/chr/cache/tmp
export TMP=/ssd/chr/cache/tmp

PYTHON_BIN="${LLMLINGUA_PYTHON:-.venv-llmlingua/bin/python}"

# device_map: default cuda; set LLMLINGUA_DEVICE=cpu when GPUs are full / torch-CUDA mismatched.
DEVICE_MAP="${LLMLINGUA_DEVICE:-cuda}"

"${PYTHON_BIN}" scripts/vllm/_serve_llmlingua2.py \
  --model-name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
  --device-map "${DEVICE_MAP}" \
  --port 8004
