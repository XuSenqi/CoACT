"""FastAPI service wrapping the official LLMLingua-2 ``PromptCompressor``.

LLMLingua-2 (arXiv:2403.12968) is a token-classification compressor
(XLM-RoBERTa-large, distilled from GPT-4). It is an encoder model and cannot
be served by vLLM, so it runs as a standalone HTTP service. The eval
framework's :class:`LLMLingua2Compression` strategy is a thin ``httpx`` client
for this service (see ``src/eval/baselines/llmlingua2.py``).

Run via ``scripts/vllm/serve_llmlingua2.sh``. Install dependencies into an
isolated virtualenv (see ``scripts/vllm/requirements-llmlingua2.txt``) because
``llmlingua`` conflicts with the ``transformers>=5`` pinned by the eval venv
(``.venv-vllm``).

The request/response contract mirrors the keys returned by the upstream
``compress_prompt`` so the strategy can forward all knobs from ``config.yaml``
unchanged.
"""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from llmlingua import PromptCompressor
from pydantic import BaseModel, Field


class CompressRequest(BaseModel):
    """Per-output compression request; defaults match the upstream signature."""

    text: str
    rate: float = 0.5
    target_token: int = -1
    force_tokens: list[str] = Field(default_factory=list)
    force_reserve_digit: bool = False
    drop_consecutive: bool = False
    chunk_end_tokens: list[str] = Field(default_factory=lambda: [".", "\n"])
    use_token_level_filter: bool = True
    use_context_level_filter: bool = False


class CompressResponse(BaseModel):
    """Pass-through of the relevant ``compress_prompt`` result fields."""

    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    rate: str
    ratio: str
    error_msg: str | None = None


def build_app(model_name: str, device_map: str) -> FastAPI:
    """Build the FastAPI app with a single eagerly-loaded compressor."""
    app = FastAPI(title="LLMLingua-2 Compression Service")
    compressor = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device_map=device_map,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/compress", response_model=CompressResponse)
    def compress(req: CompressRequest) -> CompressResponse:
        result = compressor.compress_prompt(
            req.text,
            rate=req.rate,
            target_token=req.target_token,
            force_tokens=list(req.force_tokens),
            force_reserve_digit=req.force_reserve_digit,
            drop_consecutive=req.drop_consecutive,
            chunk_end_tokens=list(req.chunk_end_tokens),
            use_token_level_filter=req.use_token_level_filter,
            use_context_level_filter=req.use_context_level_filter,
        )
        return CompressResponse(
            compressed_prompt=result["compressed_prompt"],
            origin_tokens=int(result.get("origin_tokens", 0)),
            compressed_tokens=int(result.get("compressed_tokens", 0)),
            rate=str(result.get("rate", "")),
            ratio=str(result.get("ratio", "")),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LLMLingua-2 compression over HTTP")
    parser.add_argument(
        "--model-name",
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    )
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()

    print(
        f"Loading LLMLingua-2 model: {args.model_name} (device_map={args.device_map})",
        flush=True,
    )
    app = build_app(args.model_name, args.device_map)
    print(f"Starting LLMLingua-2 service on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
