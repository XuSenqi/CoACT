"""CoACT HTTP prune service core logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from src.compression.output_parser import CompressionOutputResolution, interpret_compression_response
from src.compression.prompt_context import number_prompt_lines
from src.config.prompts import render_compression_prompt

CompressionType = Literal["unchanged", "plain", "code", "invalid"]
_COMPRESSED_LINE_PREFIX = "(compressed "


class Tokenizer(Protocol):
    """Minimal tokenizer interface for token counting."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = ...,
        return_attention_mask: bool = ...,
    ) -> dict[str, Any]: ...


class PruneRequest(BaseModel):
    """HTTP request body for POST /prune."""

    query: str
    code: str
    goal: str = ""
    tool_call: str | None = None


class PruneResponse(BaseModel):
    """HTTP response body for POST /prune."""

    pruned_code: str
    origin_token_cnt: int
    left_token_cnt: int
    model_input_token_cnt: int
    kept_frags: list[int] = Field(default_factory=list)
    error_msg: str | None = None
    compression_type: CompressionType


@dataclass(frozen=True)
class CoACTBackendConfig:
    """vLLM backend and sampling settings for the HTTP prune service."""

    backend: str
    model: str = "CoACT"
    temperature: float = 0.6
    top_p: float | None = 0.95
    top_k: int | None = 20
    min_p: float | None = 0.0
    presence_penalty: float | None = 0.0
    repetition_penalty: float | None = 1.0
    max_tokens: int = 81920
    timeout: float = 120.0


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    """Count tokens in text using the CoACT tokenizer."""
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def check_backend_reachable(backend: str, *, timeout: float = 5.0) -> bool:
    """Return whether the vLLM backend responds to GET /models."""
    models_url = f"{backend.rstrip('/')}/models"
    try:
        response = httpx.get(models_url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    return True


def _format_tool_call(tool_call: str | None) -> tuple[str, ...] | None:
    if tool_call is None:
        return None
    normalized = tool_call.strip()
    if not normalized:
        return None
    return (normalized,)


def _compression_type(resolution: CompressionOutputResolution) -> CompressionType:
    if not resolution.valid_response:
        return "invalid"
    if resolution.keep_original:
        return "unchanged"
    if resolution.used_line_references:
        return "code"
    return "plain"


def compute_kept_frags(
    original_code: str,
    pruned_code: str,
    compression_type: CompressionType,
) -> list[int]:
    """Estimate 1-indexed kept line numbers from original and pruned text."""
    original_lines = original_code.splitlines()
    if not original_lines:
        return []

    if compression_type == "unchanged":
        return list(range(1, len(original_lines) + 1))

    if compression_type == "plain":
        return []

    pruned_lines = [
        line
        for line in pruned_code.splitlines()
        if not line.startswith(_COMPRESSED_LINE_PREFIX)
    ]
    kept: list[int] = []
    pruned_index = 0
    for line_number, line in enumerate(original_lines, start=1):
        if pruned_index < len(pruned_lines) and pruned_lines[pruned_index] == line:
            kept.append(line_number)
            pruned_index += 1
    return kept


def _build_chat_completion_payload(
    *,
    config: CoACTBackendConfig,
    prompt: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    optional_fields = {
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "presence_penalty": config.presence_penalty,
        "repetition_penalty": config.repetition_penalty,
    }
    payload.update({key: value for key, value in optional_fields.items() if value is not None})
    return payload


def _error_response(
    *,
    code: str,
    tokenizer: Tokenizer,
    error_msg: str,
) -> PruneResponse:
    token_count = count_tokens(code, tokenizer)
    return PruneResponse(
        pruned_code=code,
        origin_token_cnt=token_count,
        left_token_cnt=token_count,
        model_input_token_cnt=0,
        kept_frags=list(range(1, len(code.splitlines()) + 1)) if code else [],
        error_msg=error_msg,
        compression_type="invalid",
    )


def prune_code(
    request: PruneRequest,
    *,
    config: CoACTBackendConfig,
    tokenizer: Tokenizer,
    client: httpx.Client | None = None,
) -> PruneResponse:
    """Run one CoACT compression request against the vLLM backend."""
    original_code = request.code
    numbered_output = number_prompt_lines(original_code)
    prompt = render_compression_prompt(
        goal=request.goal,
        context_focus_question=request.query,
        tool_output=numbered_output,
        tool_call=_format_tool_call(request.tool_call),
    )
    origin_token_cnt = count_tokens(original_code, tokenizer)
    completion_url = f"{config.backend.rstrip('/')}/chat/completions"
    payload = _build_chat_completion_payload(config=config, prompt=prompt)

    owns_client = client is None
    http_client = client or httpx.Client(timeout=config.timeout)
    try:
        response = http_client.post(completion_url, json=payload)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        return _error_response(
            code=original_code,
            tokenizer=tokenizer,
            error_msg=str(exc),
        )
    finally:
        if owns_client:
            http_client.close()

    try:
        raw_output = body["choices"][0]["message"]["content"]
        model_input_token_cnt = int(body.get("usage", {}).get("prompt_tokens", 0))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return _error_response(
            code=original_code,
            tokenizer=tokenizer,
            error_msg=f"Invalid vLLM response: {exc}",
        )

    resolution = interpret_compression_response(
        raw_output,
        original_output=original_code,
        numbered_output=numbered_output,
    )
    compression_type = _compression_type(resolution)
    pruned_code = resolution.effective_output
    left_token_cnt = count_tokens(pruned_code, tokenizer)

    return PruneResponse(
        pruned_code=pruned_code,
        origin_token_cnt=origin_token_cnt,
        left_token_cnt=left_token_cnt,
        model_input_token_cnt=model_input_token_cnt,
        kept_frags=compute_kept_frags(original_code, pruned_code, compression_type),
        error_msg=None,
        compression_type=compression_type,
    )
