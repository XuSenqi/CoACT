"""FastAPI HTTP wrapper for CoACT compression via a vLLM backend.

This service exposes SWE-Pruner-style endpoints (``GET /health``,
``POST /prune``) while delegating inference to an existing vLLM OpenAI
compatible server started by ``start_CoACT.sh``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from src.serving.coact_prune import (
    CoACTBackendConfig,
    PruneRequest,
    PruneResponse,
    check_backend_reachable,
    prune_code,
)


def build_app(
    *,
    backend_config: CoACTBackendConfig,
    tokenizer_path: str,
) -> FastAPI:
    """Build the FastAPI app with a loaded tokenizer and shared HTTP client."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    http_client = httpx.Client(timeout=backend_config.timeout)
    app = FastAPI(title="CoACT Compression Service")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        http_client.close()

    @app.get("/health")
    def health() -> dict[str, object]:
        backend_reachable = check_backend_reachable(backend_config.backend)
        return {
            "status": "healthy",
            "backend_reachable": backend_reachable,
            "backend_endpoint": backend_config.backend,
        }

    @app.post("/prune", response_model=PruneResponse)
    def prune(request: PruneRequest) -> PruneResponse:
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must be non-empty")
        if not request.code:
            raise HTTPException(status_code=422, detail="code must be non-empty")
        return prune_code(
            request,
            config=backend_config,
            tokenizer=tokenizer,
            client=http_client,
        )

    return app


def _env_or_default(name: str, default: str) -> str:
    return os.getenv(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CoACT compression over HTTP")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(_env_or_default("COACT_HTTP_PORT", "8002")))
    parser.add_argument(
        "--backend",
        default=_env_or_default("COACT_HTTP_BACKEND", "http://localhost:8001/v1"),
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("COACT_HTTP_MODEL", "CoACT"),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=_env_or_default("COACT_HTTP_TOKENIZER_PATH", "checkpoints/CoACT"),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=81920)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer path not found: {tokenizer_path}")

    backend_config = CoACTBackendConfig(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    print(f"Starting CoACT HTTP service on {args.host}:{args.port}", flush=True)
    print(f"vLLM backend: {backend_config.backend}", flush=True)
    print(f"Model: {backend_config.model}", flush=True)
    print(f"Tokenizer: {tokenizer_path}", flush=True)

    app = build_app(backend_config=backend_config, tokenizer_path=str(tokenizer_path))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
