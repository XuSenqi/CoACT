"""LongCodeZip two-stage, query-aware code-compression-service baseline."""

import time
from typing import Any

import httpx

from src.config.config import Config

from .base import (
    CompressionAttempt,
    CompressionStrategy,
    _usage_payload,
)

# Fixed instruction handed to LongCodeZip alongside the per-step query (the
# agent's context_focus_question). The query carries the real relevance signal
# driving the Approximated-Mutual-Information ranking; the instruction only
# frames the conditioning, so a single neutral SWE-task string is enough.
_DEFAULT_INSTRUCTION = (
    "You are solving a software engineering task. "
    "Keep the parts of the following code/tool output relevant to the task."
)


def _join_focus_questions(focus_questions: Any) -> str:
    """Collapse the per-action focus-question tuple into one query string."""
    if isinstance(focus_questions, str):
        return focus_questions.strip()
    if not focus_questions:
        return ""
    parts = [q.strip() for q in focus_questions if isinstance(q, str) and q.strip()]
    return "\n".join(parts)


class LongCodeZipCompression(CompressionStrategy):
    """LongCodeZip two-stage code-compression baseline.

    LongCodeZip (Shi et al., 2025; arXiv:2510.00446) is a *training-free,
    query-aware* code compressor. Stage 1 splits the code along function/class
    boundaries and ranks chunks by conditional perplexity (Approximated Mutual
    Information, ``AMI(c, q) = PPL(q) - PPL(q | c)``) relative to the query,
    keeping only the most relevant functions under a coarse budget. Stage 2
    detects perplexity-based blocks inside the kept functions and selects them
    with a 0/1 knapsack under an adaptive, importance-biased token budget.

    Computing perplexity needs a causal code LM, so LongCodeZip cannot be
    served by vLLM; it runs as a standalone FastAPI service
    (``scripts/vllm/serve_longcodezip.sh``) wrapping the official
    ``longcodezip.LongCodeZip`` class. This strategy is just the HTTP client;
    it only depends on ``httpx``.

    Being query-aware, it consumes the agent's ``context_focus_question`` as
    the query (``consumes_cfq = True``, like SWEPruner). The CFQ instance
    template already informs the agent that its outputs are compressed, so the
    transparent-compression notice is left off (``announces_compression`` would
    be ignored when ``cfq_enabled`` is True anyway).

    On any service/network failure the original tool output is returned
    unchanged (fail-soft, like SWEPruner / LLMLingua-2) so a single bad call
    never aborts a whole trajectory; the failure is recorded in the attempt's
    ``extras``.
    """

    name = "longcodezip"
    consumes_cfq = True

    def __init__(
        self,
        endpoint: str,
        rate: float = 0.5,
        language: str = "python",
        dynamic_compression_ratio: float = 0.2,
        context_budget: str = "+100",
        rank_only: bool = False,
        fine_ratio: float | None = None,
        fine_grained_importance_method: str = "conditional_ppl",
        min_lines_for_fine_grained: int = 5,
        importance_beta: float = 0.5,
        use_knapsack: bool = True,
        instruction: str = _DEFAULT_INSTRUCTION,
        request_timeout: float = 120.0,
        retries: int = 3,
        skip_compression_max_tokens: int = 512,
    ):
        """Initialize the LongCodeZip client.

        Args:
            endpoint: Base URL of the LongCodeZip service. ``/compress`` is
                appended for inference and ``/health`` is the readiness probe.
            rate: Target token *retention* fraction in (0, 1]; LongCodeZip keeps
                roughly this fraction of tokens.
            language: Source language passed to function-boundary splitting.
            dynamic_compression_ratio: Spread of the importance-biased per-function
                budget around the base ratio (upstream default 0.2).
            context_budget: Coarse function-budget expression forwarded verbatim
                to the upstream selector (e.g. ``"+100"``).
            rank_only: When True, run only Stage 1 (coarse function selection);
                when False (default), run the full two-stage compression.
            fine_ratio: Explicit Stage-2 retention ratio; ``None`` derives it
                from ``rate`` (upstream default).
            fine_grained_importance_method: Stage-2 block importance metric,
                ``"conditional_ppl"`` (default) or ``"contrastive_perplexity"``.
            min_lines_for_fine_grained: Functions with fewer lines are kept
                intact instead of being block-pruned (upstream default 5).
            importance_beta: Sensitivity of per-function budget to normalized
                importance (upstream default 0.5).
            use_knapsack: Use 0/1-knapsack block selection (Stage 2) when True.
            instruction: Fixed instruction framing the conditional-perplexity
                ranking; the query comes from the agent's focus question.
            request_timeout: Per-request HTTP timeout in seconds.
            retries: Number of HTTP attempts (including the first). When all
                attempts fail the original output is returned unchanged.
            skip_compression_max_tokens: Skip compression entirely when the raw
                tool output is at or below this many characters, matching the
                short-output bypass used by the other baselines.
        """
        self.endpoint = endpoint.rstrip("/")
        self.rate = rate
        self.language = language
        self.dynamic_compression_ratio = dynamic_compression_ratio
        self.context_budget = context_budget
        self.rank_only = rank_only
        self.fine_ratio = fine_ratio
        self.fine_grained_importance_method = fine_grained_importance_method
        self.min_lines_for_fine_grained = min_lines_for_fine_grained
        self.importance_beta = importance_beta
        self.use_knapsack = use_knapsack
        self.instruction = instruction
        self.request_timeout = request_timeout
        self.retries = retries
        self.skip_compression_max_tokens = skip_compression_max_tokens
        self._client = httpx.Client(timeout=request_timeout)

    @classmethod
    def from_config(cls, config: Config) -> "LongCodeZipCompression":
        return cls(
            endpoint=config.evaluation.longcodezip.endpoint,
            rate=config.evaluation.longcodezip.rate,
            language=config.evaluation.longcodezip.language,
            dynamic_compression_ratio=config.evaluation.longcodezip.dynamic_compression_ratio,
            context_budget=config.evaluation.longcodezip.context_budget,
            rank_only=config.evaluation.longcodezip.rank_only,
            fine_ratio=config.evaluation.longcodezip.fine_ratio,
            fine_grained_importance_method=(
                config.evaluation.longcodezip.fine_grained_importance_method
            ),
            min_lines_for_fine_grained=config.evaluation.longcodezip.min_lines_for_fine_grained,
            importance_beta=config.evaluation.longcodezip.importance_beta,
            use_knapsack=config.evaluation.longcodezip.use_knapsack,
            request_timeout=config.evaluation.longcodezip.request_timeout,
            retries=config.evaluation.longcodezip.retries,
            skip_compression_max_tokens=config.evaluation.longcodezip.skip_compression_max_tokens,
        )

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """Compress a tool output via the LongCodeZip service.

        Args:
            tool_output: Raw tool output text (treated as the ``code`` context).
            context: Compression context; we only consume ``context_focus_question``
                as the query.

        Returns:
            Tuple of ``(compressed_code, attempt)``. ``attempt`` is ``None``
            only when the strategy short-circuits before any HTTP call (no focus
            question, or short-output bypass). On service failure the original
            output is returned with an attempt whose ``extras`` carry
            ``error_msg``.
        """
        if context is None:
            raise ValueError("Compression context is required for LongCodeZip")

        query = _join_focus_questions(context["context_focus_question"])
        if not query:
            return tool_output, None

        if len(tool_output) <= self.skip_compression_max_tokens:
            return tool_output, None

        payload = {
            "code": tool_output,
            "query": query,
            "instruction": self.instruction,
            "rate": self.rate,
            "language": self.language,
            "dynamic_compression_ratio": self.dynamic_compression_ratio,
            "context_budget": self.context_budget,
            "rank_only": self.rank_only,
            "fine_ratio": self.fine_ratio,
            "fine_grained_importance_method": self.fine_grained_importance_method,
            "min_lines_for_fine_grained": self.min_lines_for_fine_grained,
            "importance_beta": self.importance_beta,
            "use_knapsack": self.use_knapsack,
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

        compressed_code = body["compressed_code"]
        origin_tokens = int(body.get("original_tokens", 0))
        compressed_tokens = int(body.get("compressed_tokens", 0))

        attempt = CompressionAttempt(
            time_ms=elapsed_ms,
            usage=_usage_payload(
                prompt_tokens=origin_tokens,
                completion_tokens=compressed_tokens,
            ),
            extras={
                "model": self.name,
                "original_tokens": origin_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": body.get("compression_ratio", ""),
                "rank_only": self.rank_only,
            },
        )
        return compressed_code, attempt
