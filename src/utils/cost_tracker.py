"""Cost tracking utilities backed by the project's own ``PricingConfig``."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.config.config import ModelPricingEntry, PricingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostBreakdown:
    """Per-category USD cost for one LiteLLM API call."""

    input_usd: float         # Non-cached input tokens × input_cost_per_token
    cache_read_usd: float    # Cache-hit input tokens × cache_read_input_token_cost
    cache_creation_usd: float  # Cache-write tokens × cache_creation_input_token_cost (Anthropic)
    output_usd: float        # Completion tokens × output_cost_per_token
    total_usd: float         # Sum of all four categories


_ZERO_BREAKDOWN = CostBreakdown(
    input_usd=0.0,
    cache_read_usd=0.0,
    cache_creation_usd=0.0,
    output_usd=0.0,
    total_usd=0.0,
)


_active_pricing: PricingConfig = PricingConfig()


def set_active_pricing(pricing: PricingConfig) -> None:
    """Install ``pricing`` as the active per-token pricing source.

    Subsequent calls to :func:`get_cost_breakdown_from_usage` resolve model
    rates through this config. Call once during runner / script bootstrap.
    """
    global _active_pricing
    _active_pricing = pricing


def extract_response_usage(response: Any) -> dict[str, int] | None:
    """Extract normalized usage fields from a serialized LiteLLM response."""
    if not isinstance(response, Mapping):
        return None

    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)

    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(prompt_details, Mapping):
        cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)

    cache_creation_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
    }


def _breakdown_for_entry(
    entry: ModelPricingEntry,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cache_creation_tokens: int,
) -> CostBreakdown:
    non_cached = max(prompt_tokens - cached_tokens - cache_creation_tokens, 0)
    input_usd = non_cached * entry.input_cost_per_token
    cache_read_usd = cached_tokens * entry.cache_read_input_token_cost
    cache_creation_usd = cache_creation_tokens * entry.cache_creation_input_token_cost
    output_usd = completion_tokens * entry.output_cost_per_token
    return CostBreakdown(
        input_usd=input_usd,
        cache_read_usd=cache_read_usd,
        cache_creation_usd=cache_creation_usd,
        output_usd=output_usd,
        total_usd=input_usd + cache_read_usd + cache_creation_usd + output_usd,
    )


def get_cost_breakdown_from_usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    pricing: PricingConfig | None = None,
) -> CostBreakdown:
    """Compute cost from normalized usage fields against ``PricingConfig``.

    Args:
        model: Model identifier exactly as registered in ``PricingConfig``.
        pricing: Optional override; defaults to the module-level active config
            installed via :func:`set_active_pricing`. Useful for tests and for
            offline recomputation against an arbitrary pricing snapshot.
    """
    config = pricing if pricing is not None else _active_pricing
    entry = config.get_pricing(model)
    return _breakdown_for_entry(
        entry,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def get_response_cost_breakdown(
    response: dict[str, Any],
    model: str,
    *,
    pricing: PricingConfig | None = None,
) -> CostBreakdown:
    """Compute per-category USD cost for one serialized LiteLLM response dict.

    Reads token counts from usage, looks up the rate via ``PricingConfig``,
    and returns a breakdown by cost category. Returns all-zero when usage is
    missing.

    Args:
        response: Serialized LiteLLM response dict (as stored in trajectory extra).
        model: Model identifier used for the call.
        pricing: Optional pricing override (see :func:`get_cost_breakdown_from_usage`).
    """
    usage = extract_response_usage(response)
    if usage is None:
        return _ZERO_BREAKDOWN

    return get_cost_breakdown_from_usage(
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cached_tokens=usage["cached_tokens"],
        cache_creation_tokens=usage["cache_creation_input_tokens"],
        pricing=pricing,
    )
