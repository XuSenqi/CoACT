#!/bin/bash
# vLLM serve script for Agent model (Qwen3.5-35B-A3B-FP8)
# Used as the main agent that processes compressed tool outputs

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

export VLLM_CACHE_ROOT=/ssd/chr/vllm_cache
export HF_HOME=/ssd/chr/vllm_cache/huggingface
export TORCH_HOME=/ssd/chr/vllm_cache/torch
export TMPDIR=/ssd/chr/cache/tmp
export TEMP=/ssd/chr/cache/tmp
export TMP=/ssd/chr/cache/tmp
export TRITON_CACHE_DIR=/ssd/chr/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ssd/chr/cache/torchinductor

vllm serve model/Qwen3.5-35B-A3B-FP8 \
  --port 8000 \
  --served-model-name Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  -dp 4 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.95 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-prompt-tokens-details \
  --language-model-only