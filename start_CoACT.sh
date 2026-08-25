#!/bin/bash

set -euo pipefail

cd /data/CoACT

unset LD_LIBRARY_PATH
export VLLM_NVIDIA_LIBS=/data/CoACT/.venv-vllm/lib/python3.13/site-packages
export LD_LIBRARY_PATH=$VLLM_NVIDIA_LIBS/nvidia/nvjitlink/lib:$VLLM_NVIDIA_LIBS/nvidia/cusparse/lib:$VLLM_NVIDIA_LIBS/torch/lib


export CUDA_VISIBLE_DEVICES=0   # 若只有一张卡且 Agent 已占满，见下方说明

#nohup ./scripts/run_vllm.sh serve checkpoints/CoACT \
#  --host 0.0.0.0 \
#  --port 8001 \
#  --served-model-name CoACT \
#  --tensor-parallel-size 1 \
#  --max-model-len 131072 \
#  --reasoning-parser qwen3 \
#  --gpu-memory-utilization 0.20 \
#  --language-model-only \
#  --gdn-prefill-backend triton >> CoACT.log &

nohup ./scripts/run_vllm.sh serve checkpoints/CoACT \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name CoACT \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.20 \
  --language-model-only \
  --gdn-prefill-backend triton >> CoACT.log &
