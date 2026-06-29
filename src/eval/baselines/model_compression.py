"""Shared model-backed compression strategy implementation."""

import time
from typing import Any

from litellm import completion

from src.config.prompts import render_compression_prompt

from .base import (
    CompressionAttempt,
    CompressionStrategy,
    _build_completion_kwargs,
    _price_litellm_response,
)


class _ModelCompressionStrategy(CompressionStrategy):
    """Base class for strategies that compress observations with a chat model."""

    consumes_cfq = True

    def __init__(
        self,
        model: str,
        api_endpoint: str,
        temperature: float = 0.6,
        top_p: float | None = 0.95,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        max_tokens: int = 81920,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize model-backed compression settings.

        Args:
            model: LiteLLM model identifier.
            api_endpoint: API base URL for the model.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            min_p: Minimum probability parameter.
            presence_penalty: Presence penalty.
            repetition_penalty: Repetition penalty.
            max_tokens: Maximum tokens in compressed output.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum LiteLLM retry attempts.
        """
        self.model = model
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """Compress tool output with the configured model.

        Args:
            tool_output: The tool output to compress.
            context: Runtime compression context containing ``goal``,
                ``context_focus_question``, ``tool_output_for_prompt``, and
                ``tool_call``.

        Returns:
            Tuple of compressed output and a priced compression attempt.

        Raises:
            ValueError: If context is missing.
        """
        if context is None:
            raise ValueError("Compression context is required for model-based strategies")

        start_time = time.perf_counter()
        prompt = render_compression_prompt(
            goal=context["goal"],
            context_focus_question=context["context_focus_question"],
            tool_output=context["tool_output_for_prompt"],
            tool_call=context["tool_call"],
        )

        kwargs = _build_completion_kwargs(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_endpoint=self.api_endpoint,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        response = completion(**kwargs)
        compressed = response.choices[0].message.content

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        attempt = _price_litellm_response(
            model=self.model,
            response=response,
            time_ms=elapsed_ms,
        )
        return compressed, attempt
