"""LLMLingua-2 task-agnostic compression-service baseline."""

import time
from typing import Any

import httpx

from src.config.config import Config

from .base import (
    CompressionAttempt,
    CompressionStrategy,
    _usage_payload,
)


class LLMLingua2Compression(CompressionStrategy):
    """LLMLingua-2 token-classification compression baseline.

    LLMLingua-2 (Pan et al., 2024; arXiv:2403.12968) is a *task-agnostic*
    prompt compressor: an XLM-RoBERTa-large token classifier distilled from
    GPT-4 that predicts a preserve/discard label per token. Because it is an
    encoder model it cannot be served by vLLM, so it runs as a standalone
    FastAPI service (``scripts/vllm/serve_llmlingua2.sh``) wrapping the
    official ``llmlingua.PromptCompressor``. This strategy is just the HTTP
    client; it only depends on ``httpx``.

    Being task-agnostic, it ignores the agent's ``context_focus_question``
    (``consumes_cfq = False``) and compresses each tool output to a fixed
    ``rate`` (or ``target_token`` budget). All compression knobs are forwarded
    from config so ``config.yaml`` stays the single source of truth, and their
    defaults mirror the upstream ``compress_prompt_llmlingua2`` signature.

    On any service/network failure the original tool output is returned
    unchanged (fail-soft, like SWEPruner) so a single bad call never aborts a
    whole trajectory; the failure is recorded in the attempt's ``extras``.
    """

    name = "llmlingua2"
    consumes_cfq = False
    announces_compression = True

    def __init__(
        self,
        endpoint: str,
        rate: float = 0.8,
        target_token: int = -1,
        force_tokens: tuple[str, ...] = (),
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
        chunk_end_tokens: tuple[str, ...] = (".", "\n"),
        use_token_level_filter: bool = True,
        use_context_level_filter: bool = False,
        request_timeout: float = 120.0,
        retries: int = 3,
        skip_compression_max_tokens: int = 512,
    ):
        """Initialize the LLMLingua-2 client.

        Args:
            endpoint: Base URL of the LLMLingua-2 service. ``/compress`` is
                appended for inference and ``/health`` is the readiness probe.
            rate: Target retention ratio in (0, 1]; LLMLingua-2 keeps roughly
                this fraction of tokens. Ignored when ``target_token`` >= 0.
            target_token: Absolute post-compression token budget; ``-1``
                disables it and uses ``rate`` instead.
            force_tokens: Tokens always preserved (upstream default is empty).
            force_reserve_digit: Force-keep tokens containing a digit.
            drop_consecutive: Drop ``force_tokens`` that appear consecutively.
            chunk_end_tokens: Early-stop tokens for chunk segmentation.
            use_token_level_filter: Enable token-level filtering.
            use_context_level_filter: Enable context-level filtering (off for
                single tool outputs).
            request_timeout: Per-request HTTP timeout in seconds.
            retries: Number of HTTP attempts (including the first). When all
                attempts fail the original output is returned unchanged.
            skip_compression_max_tokens: Skip compression entirely when the raw
                tool output is at or below this many characters, matching the
                short-output bypass used by the other baselines.
        """
        self.endpoint = endpoint.rstrip("/")
        self.rate = rate
        self.target_token = target_token
        self.force_tokens = list(force_tokens)
        self.force_reserve_digit = force_reserve_digit
        self.drop_consecutive = drop_consecutive
        self.chunk_end_tokens = list(chunk_end_tokens)
        self.use_token_level_filter = use_token_level_filter
        self.use_context_level_filter = use_context_level_filter
        self.request_timeout = request_timeout
        self.retries = retries
        self.skip_compression_max_tokens = skip_compression_max_tokens
        self._client = httpx.Client(timeout=request_timeout)

    @classmethod
    def from_config(cls, config: Config) -> "LLMLingua2Compression":
        return cls(
            endpoint=config.evaluation.llmlingua2.endpoint,
            rate=config.evaluation.llmlingua2.rate,
            target_token=config.evaluation.llmlingua2.target_token,
            force_tokens=config.evaluation.llmlingua2.force_tokens,
            force_reserve_digit=config.evaluation.llmlingua2.force_reserve_digit,
            drop_consecutive=config.evaluation.llmlingua2.drop_consecutive,
            chunk_end_tokens=config.evaluation.llmlingua2.chunk_end_tokens,
            use_token_level_filter=config.evaluation.llmlingua2.use_token_level_filter,
            use_context_level_filter=config.evaluation.llmlingua2.use_context_level_filter,
            request_timeout=config.evaluation.llmlingua2.request_timeout,
            retries=config.evaluation.llmlingua2.retries,
            skip_compression_max_tokens=config.evaluation.llmlingua2.skip_compression_max_tokens,
        )

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """Compress a tool output via the LLMLingua-2 service.

        Args:
            tool_output: Raw tool output text.
            context: Unused; LLMLingua-2 is task-agnostic and ignores the
                focus question. Accepted to satisfy the strategy interface.

        Returns:
            Tuple of ``(compressed_text, attempt)``. ``attempt`` is ``None``
            only when the strategy short-circuits before any HTTP call
            (short-output bypass). On service failure the original output is
            returned with an attempt whose ``extras`` carry ``error_msg``.
        """
        if len(tool_output) <= self.skip_compression_max_tokens:
            return tool_output, None

        payload = {
            "text": tool_output,
            "rate": self.rate,
            "target_token": self.target_token,
            "force_tokens": self.force_tokens,
            "force_reserve_digit": self.force_reserve_digit,
            "drop_consecutive": self.drop_consecutive,
            "chunk_end_tokens": self.chunk_end_tokens,
            "use_token_level_filter": self.use_token_level_filter,
            "use_context_level_filter": self.use_context_level_filter,
        }

        start_time = time.perf_counter()
        body: dict[str, Any] | None = None
        last_error: Exception | None = None
        for _ in range(max(self.retries, 1)):
            try:
                response = self._client.post(f"{self.endpoint}/compress", json=payload)
                response.raise_for_status()
                body = response.json()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Retries exhausted: fail soft, return the original output unchanged.
        if body is None:
            assert last_error is not None
            return (
                tool_output,
                CompressionAttempt(
                    time_ms=elapsed_ms,
                    extras={"model": self.name, "error_msg": str(last_error)},
                ),
            )

        # Service returned 200 but flagged an error in the payload.
        error_msg = body.get("error_msg")
        if error_msg:
            return (
                tool_output,
                CompressionAttempt(
                    time_ms=elapsed_ms,
                    extras={"model": self.name, "error_msg": error_msg},
                ),
            )

        compressed_prompt = body["compressed_prompt"]
        origin_tokens = int(body.get("origin_tokens", 0))
        compressed_tokens = int(body.get("compressed_tokens", 0))

        attempt = CompressionAttempt(
            time_ms=elapsed_ms,
            usage=_usage_payload(
                prompt_tokens=origin_tokens,
                completion_tokens=compressed_tokens,
            ),
            extras={
                "model": self.name,
                "origin_tokens": origin_tokens,
                "compressed_tokens": compressed_tokens,
                "ratio": body.get("ratio", ""),
                "rate": body.get("rate", ""),
            },
        )
        return compressed_prompt, attempt
