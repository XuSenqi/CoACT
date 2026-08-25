#!/bin/bash

set -euo pipefail

#nohup .venv-vllm/bin/vllm serve model/Qwen3.5-35B-A3B-FP8   --host 0.0.0.0   --port 8000   --served-model-name Qwen3.5-35B-A3B-FP8   --tensor-parallel-size 1   --max-model-len 131072   --reasoning-parser qwen3   --enable-auto-tool-choice   --tool-call-parser qwen3_coder   --gpu-memory-utilization 0.90   --language-model-only >> Qwen.log &

#nohup .venv-vllm/bin/vllm serve model/Qwen3.5-35B-A3B-FP8   --host 0.0.0.0   --port 8000   --served-model-name Qwen3.5-35B-A3B-FP8   --tensor-parallel-size 1   --max-model-len 65536  --reasoning-parser qwen3   --enable-auto-tool-choice   --tool-call-parser qwen3_coder   --gpu-memory-utilization 0.70 --language-model-only >> Qwen.log &

unset LD_LIBRARY_PATH
export VLLM_NVIDIA_LIBS=/data/CoACT/.venv-vllm/lib/python3.13/site-packages
export LD_LIBRARY_PATH=$VLLM_NVIDIA_LIBS/nvidia/nvjitlink/lib:$VLLM_NVIDIA_LIBS/nvidia/cusparse/lib:$VLLM_NVIDIA_LIBS/torch/lib

export CUDA_VISIBLE_DEVICES=0

nohup ./scripts/run_vllm.sh serve model/Qwen3.5-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.70 \
  --language-model-only \
  --gdn-prefill-backend triton >> Qwen.log &
