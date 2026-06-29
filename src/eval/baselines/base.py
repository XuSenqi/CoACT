"""Shared types and helpers for compression strategies.

Every strategy returns a :class:`CompressionAttempt` (or ``None``) from its
``compress()`` method. The agent-side accumulator is a pure adder — strategies
own pricing themselves. LiteLLM-backed strategies share
:func:`_price_litellm_response`; local-inference strategies fill ``cost``
with zeros and ``usage`` with their own measurements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.utils.cost_tracker import (
    extract_response_usage,
    get_cost_breakdown_from_usage,
)


def _zero_usage_payload() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 0,
    }


def _zero_cost_payload() -> dict[str, float]:
    return {
        "input_cost_usd": 0.0,
        "cache_read_cost_usd": 0.0,
        "cache_creation_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }


def _usage_payload(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cached_tokens": int(cached_tokens),
        "cache_creation_input_tokens": int(cache_creation_input_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
    }


@dataclass(frozen=True)
class CompressionAttempt:
    """One billable compression call.

    Strategies build this themselves and hand it back to the agent. The
    accumulator is then a pure adder — no dispatch on shape, no internal
    pricing logic. Local-inference strategies pass zero ``cost`` with real
    ``usage`` and ``time_ms``; LiteLLM-backed strategies use
    :func:`_price_litellm_response` to fill all three at once.

    Attributes:
        time_ms: Wall-clock time consumed by this compression call.
        cost: Monetary cost breakdown in USD (zero-filled by default).
        usage: Token usage breakdown (zero-filled by default).
        extras: Strategy-specific extras (e.g. SWEPruner ``score`` /
            ``kept_frags``) carried for inspection / trajectory logging
            but never aggregated into ``compression_stats``.
    """

    time_ms: float
    cost: dict[str, float] = field(default_factory=_zero_cost_payload)
    usage: dict[str, int] = field(default_factory=_zero_usage_payload)
    extras: dict[str, Any] = field(default_factory=dict)


def _price_litellm_response(
    *,
    model: str,
    response: Any,
    time_ms: float,
) -> CompressionAttempt:
    """Build a CompressionAttempt from a LiteLLM response.

    Raises:
        RuntimeError: If the response carries no usable usage block —
            silent zeros would hide a real provider regression.
    """
    serialized = response.model_dump() if hasattr(response, "model_dump") else response
    extracted = extract_response_usage(serialized)
    if extracted is None:
        raise RuntimeError(
            f"LiteLLM response from {model} is missing usage; cannot price compression call"
        )
    breakdown = get_cost_breakdown_from_usage(
        model=model,
        prompt_tokens=extracted["prompt_tokens"],
        completion_tokens=extracted["completion_tokens"],
        cached_tokens=extracted["cached_tokens"],
        cache_creation_tokens=extracted["cache_creation_input_tokens"],
    )
    return CompressionAttempt(
        time_ms=time_ms,
        cost={
            "input_cost_usd": breakdown.input_usd,
            "cache_read_cost_usd": breakdown.cache_read_usd,
            "cache_creation_cost_usd": breakdown.cache_creation_usd,
            "output_cost_usd": breakdown.output_usd,
            "total_cost_usd": breakdown.total_usd,
        },
        usage=_usage_payload(
            prompt_tokens=extracted["prompt_tokens"],
            completion_tokens=extracted["completion_tokens"],
            cached_tokens=extracted["cached_tokens"],
            cache_creation_input_tokens=extracted["cache_creation_input_tokens"],
        ),
    )


def _build_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    api_endpoint: str,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Build LiteLLM kwargs, dropping None sampling params so unset knobs fall through."""
    kwargs: dict[str, Any] = {
        k: v
        for k, v in {
            "model": model,
            "messages": messages,
            "drop_params": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "api_base": api_endpoint,
            "timeout": timeout,
            "max_retries": max_retries,
            "stop": stop,
        }.items()
        if v is not None
    }
    if "qwen3.5" in model.lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


class CompressionStrategy(ABC):
    """Base class for compression strategies."""

    name: str = "base"
    consumes_cfq: ClassVar[bool] = False
    announces_compression: ClassVar[bool] = False

    @abstractmethod
    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """Compress a tool output.

        Args:
            tool_output: The original tool output to compress.
            context: Optional context (e.g., goal, intent, step_index).

        Returns:
            Tuple of ``(compressed_output, attempt)``. ``attempt`` is
            ``None`` when the strategy short-circuits without doing any
            work (no focus question, no-op strategy). Otherwise it is a
            :class:`CompressionAttempt` reporting time / cost / usage,
            already priced by the strategy itself.
        """
        ...
