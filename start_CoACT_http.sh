#!/bin/bash

set -euo pipefail

cd /data/CoACT

# Requires vLLM CoACT service already running on port 8001 (see start_CoACT.sh).

cd /data/CoACT

unset LD_LIBRARY_PATH
export VLLM_NVIDIA_LIBS=/data/CoACT/.venv-vllm/lib/python3.13/site-packages
export LD_LIBRARY_PATH=$VLLM_NVIDIA_LIBS/nvidia/nvjitlink/lib:$VLLM_NVIDIA_LIBS/nvidia/cusparse/lib:$VLLM_NVIDIA_LIBS/torch/lib

nohup .venv-vllm/bin/python scripts/vllm/_serve_coact_http.py \
  --host 0.0.0.0 \
  --port 8002 \
  --backend http://localhost:8001/v1 \
  --model CoACT \
  --tokenizer-path checkpoints/CoACT \
  >> CoACT_http.log &

echo "CoACT HTTP service starting on port 8002 (log: CoACT_http.log)"
