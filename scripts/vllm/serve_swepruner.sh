#!/bin/bash
# Serve script for SWEPruner pruning service (fine-tuned Qwen3-Reranker)
# Used as a baseline compression strategy that prunes irrelevant lines from tool outputs

export CUDA_VISIBLE_DEVICES=4

export HF_HOME=/ssd/chr/vllm_cache/huggingface
export TORCH_HOME=/ssd/chr/vllm_cache/torch
export TMPDIR=/ssd/chr/cache/tmp
export TEMP=/ssd/chr/cache/tmp
export TMP=/ssd/chr/cache/tmp

python scripts/vllm/_serve_swepruner.py --model-path model/swepruner --port 8003
