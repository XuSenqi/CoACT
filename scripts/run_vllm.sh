#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SITE="$REPO_ROOT/.venv-vllm/lib/python3.13/site-packages"
# 避免系统旧 CUDA 库覆盖 venv 里的库
unset CUDA_HOME
export LD_LIBRARY_PATH="$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/torch/lib"
exec "$REPO_ROOT/.venv-vllm/bin/vllm" "$@"
