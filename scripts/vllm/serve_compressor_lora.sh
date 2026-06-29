#!/bin/bash
# vLLM serve script for LoRA-trained compressor model (Qwen3.5-4B)
# Used for evaluation with the trained SFT adapter

export CUDA_VISIBLE_DEVICES=0
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

vllm serve checkpoints/merged/Qwen3.5-4B-sft-merged-dsv4pro \
  --port 8010 \
  --served-model-name Qwen3.5-4B-sft-merged-dsv4pro \
  --tensor-parallel-size 1 \
  -dp 1 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.95 \
  --enable-prompt-tokens-details \
  --language-model-only