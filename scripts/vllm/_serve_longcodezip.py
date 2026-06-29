"""FastAPI service wrapping the official LongCodeZip compressor.

LongCodeZip (arXiv:2510.00446) is a training-free, query-aware, two-stage code
compressor. It ranks function-level chunks by conditional perplexity
(Approximated Mutual Information) w.r.t. the query, then prunes blocks inside
the kept functions with a 0/1 knapsack. Computing perplexity needs a causal
code LM (e.g. Qwen2.5-Coder-7B-Instruct), so LongCodeZip cannot be served by
vLLM and runs as this standalone HTTP service. The eval framework's
:class:`LongCodeZipCompression` strategy is a thin ``httpx`` client for it (see
``src/eval/baselines/longcodezip.py``).

Run via ``scripts/vllm/serve_longcodezip.sh``. Install dependencies into an
isolated virtualenv (see ``scripts/vllm/requirements-longcodezip.txt``) because
the ``longcodezip`` package pins a ``transformers`` version that conflicts with
the ``transformers>=5`` used by the eval venv (``.venv-vllm``).

The request/response contract mirrors the keys consumed by the strategy and the
upstream ``compress_code_file`` signature, so the strategy can forward every
algorithm knob from ``config.yaml`` unchanged.
"""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from longcodezip import LongCodeZip
from pydantic import BaseModel


class CompressRequest(BaseModel):
    """Per-output compression request; defaults match compress_code_file."""

    code: str
    query: str = ""
    instruction: str = ""
    rate: float = 0.5
    language: str = "python"
    dynamic_compression_ratio: float = 0.2
    context_budget: str = "+100"
    rank_only: bool = False
    fine_ratio: float | None = None
    fine_grained_importance_method: str = "conditional_ppl"
    min_lines_for_fine_grained: int = 5
    importance_beta: float = 0.5
    use_knapsack: bool = True


class CompressResponse(BaseModel):
    """Relevant fields of the upstream compress_code_file result dict."""

    compressed_code: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    error_msg: str | None = None


def build_app(model_name: str, device_map: str) -> FastAPI:
    """Build the FastAPI app with a single eagerly-loaded compressor."""
    app = FastAPI(title="LongCodeZip Compression Service")
    compressor = LongCodeZip(model_name=model_name, device_map=device_map)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/compress", response_model=CompressResponse)
    def compress(req: CompressRequest) -> CompressResponse:
        try:
            result = compressor.compress_code_file(
                code=req.code,
                query=req.query,
                instruction=req.instruction,
                rate=req.rate,
                language=req.language,
                dynamic_compression_ratio=req.dynamic_compression_ratio,
                context_budget=req.context_budget,
                rank_only=req.rank_only,
                fine_ratio=req.fine_ratio,
                fine_grained_importance_method=req.fine_grained_importance_method,
                min_lines_for_fine_grained=req.min_lines_for_fine_grained,
                importance_beta=req.importance_beta,
                use_knapsack=req.use_knapsack,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft: surface to the client.
            return CompressResponse(
                compressed_code=req.code,
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                error_msg=f"{type(exc).__name__}: {exc}",
            )

        return CompressResponse(
            compressed_code=result["compressed_code"],
            original_tokens=int(result["original_tokens"]),
            compressed_tokens=int(result["final_compressed_tokens"]),
            compression_ratio=float(result["compression_ratio"]),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LongCodeZip compression over HTTP")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    print(
        f"Loading LongCodeZip compressor: {args.model_name} (device_map={args.device_map})",
        flush=True,
    )
    app = build_app(args.model_name, args.device_map)
    print(f"Starting LongCodeZip service on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
