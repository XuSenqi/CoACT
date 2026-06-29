#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi

cd "${REPO_ROOT}"
export UV_PROJECT_ENVIRONMENT=.venv-training

exec uv sync --frozen --no-default-groups --group dev --extra training "$@"
