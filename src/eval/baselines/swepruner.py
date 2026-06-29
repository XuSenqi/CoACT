"""SWEPruner pruning-service baseline."""

import time
from typing import Any

import httpx

from src.config.config import Config

from .base import (
    CompressionAttempt,
    CompressionStrategy,
    _usage_payload,
)

_PREFIX_ALL_KEPT = "All outputs are judged as relevent! Output:\n"
_PREFIX_FILTERED = (
    "Filtered some unrelevant parts judged by your context_focus_question, "
    "good try! Filtered Output:\n"
)


def _format_pruner_error(error_msg: str, original_output: str) -> str:
    """Match the reference implementation's [Pruner Error] envelope."""
    return f"[Pruner Error]: {error_msg}\n\nOriginal Output:\n{original_output}"


class SWEPrunerCompression(CompressionStrategy):
    """SWEPruner pruning-service baseline.

    Sends each tool output to a local SWEPruner FastAPI service that runs a
    fine-tuned Qwen3-Reranker (TokenScorer). The service scores per-token
    relevance against the agent's ``context_focus_question`` and returns the
    same text with irrelevant lines collapsed into ``(filtered N lines)``
    markers. Pruning is stateless per call; we only need an HTTP client.

    The returned text always carries one of three paper-style envelopes
    so the agent receives an explicit feedback signal about what the
    pruner did with its focus question:

    * ``"All outputs are judged as relevent! Output:\\n"`` — all lines kept
    * ``"Filtered some unrelevant parts ... Filtered Output:\\n"`` — pruned
    * ``"[Pruner Error]: ...\\n\\nOriginal Output:\\n"`` — service / network failure

    The service is launched separately via ``scripts/vllm/serve_swepruner.sh``.
    """

    name = "swepruner"
    consumes_cfq = True

    def __init__(
        self,
        endpoint: str,
        threshold: float = 0.4,
        always_keep_first_frags: bool = False,
        chunk_overlap_tokens: int = 50,
        request_timeout: float = 120.0,
        retries: int = 3,
        skip_compression_max_tokens: int = 512,
    ):
        """Initialize the SWEPruner client.

        Args:
            endpoint: Base URL of the SWEPruner service. ``/prune`` is appended
                for inference and ``/health`` is the readiness probe.
            threshold: Pruning aggressiveness in [0, 1]; higher drops more lines.
            always_keep_first_frags: Forward to the service request payload.
            chunk_overlap_tokens: Per-chunk overlap when long inputs are split.
            request_timeout: Per-request HTTP timeout in seconds.
            retries: Number of HTTP attempts (including the first). On all
                attempts failing the original output is returned with the
                ``[Pruner Error]`` envelope, mirroring the paper.
            skip_compression_max_tokens: Skip pruning entirely when the raw
                tool output is at or below this many ``len(str)`` characters
                (paper's ``min_chars``; reference impl uses ``len(req.code)``).
        """
        self.endpoint = endpoint.rstrip("/")
        self.threshold = threshold
        self.always_keep_first_frags = always_keep_first_frags
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.request_timeout = request_timeout
        self.retries = retries
        self.skip_compression_max_tokens = skip_compression_max_tokens
        self._client = httpx.Client(timeout=request_timeout)

    @classmethod
    def from_config(cls, config: Config) -> "SWEPrunerCompression":
        return cls(
            endpoint=config.evaluation.swepruner.endpoint,
            threshold=config.evaluation.swepruner.threshold,
            always_keep_first_frags=config.evaluation.swepruner.always_keep_first_frags,
            chunk_overlap_tokens=config.evaluation.swepruner.chunk_overlap_tokens,
            request_timeout=config.evaluation.swepruner.request_timeout,
            retries=config.evaluation.swepruner.retries,
            skip_compression_max_tokens=config.evaluation.swepruner.skip_compression_max_tokens,
        )

    @staticmethod
    def _join_focus_questions(focus_questions: Any) -> str:
        """Collapse the per-action focus-question tuple into one query string."""
        if isinstance(focus_questions, str):
            return focus_questions.strip()
        if not focus_questions:
            return ""
        parts = [q.strip() for q in focus_questions if isinstance(q, str) and q.strip()]
        return "\n".join(parts)

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """Prune a tool output via SWEPruner.

        Args:
            tool_output: Raw tool output text (unnumbered).
            context: Compression context; we only consume ``context_focus_question``.

        Returns:
            Tuple of ``(text_with_envelope, attempt)``. ``text_with_envelope``
            carries one of the paper's three feedback prefixes. ``attempt`` is
            ``None`` only when the strategy short-circuits before issuing any
            HTTP call (no focus question).
        """
        if context is None:
            raise ValueError("Compression context is required for SWEPruner")

        query = self._join_focus_questions(context["context_focus_question"])
        if not query:
            return tool_output, None

        if len(tool_output) <= self.skip_compression_max_tokens:
            return _PREFIX_ALL_KEPT + tool_output, None

        payload = {
            "code": tool_output,
            "query": query,
            "threshold": self.threshold,
            "always_keep_first_frags": self.always_keep_first_frags,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
        }

        start_time = time.perf_counter()
        body: dict[str, Any] | None = None
        last_error: Exception | None = None
        for _ in range(max(self.retries, 1)):
            try:
                response = self._client.post(f"{self.endpoint}/prune", json=payload)
                response.raise_for_status()
                body = response.json()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Retries exhausted: fall back to the original output behind a [Pruner Error] envelope.
        if body is None:
            assert last_error is not None
            return (
                _format_pruner_error(str(last_error), tool_output),
                CompressionAttempt(
                    time_ms=elapsed_ms,
                    extras={"model": self.name, "error_msg": str(last_error)},
                ),
            )

        # Service returned 200 but flagged an error in the payload (e.g. query too long).
        error_msg = body.get("error_msg")
        if error_msg:
            return (
                _format_pruner_error(error_msg, tool_output),
                CompressionAttempt(
                    time_ms=elapsed_ms,
                    extras={"model": self.name, "error_msg": error_msg},
                ),
            )

        pruned_code = body["pruned_code"]
        origin_tokens = int(body.get("origin_token_cnt", 0))
        left_tokens = int(body.get("left_token_cnt", 0))

        if left_tokens == origin_tokens:
            delivered = _PREFIX_ALL_KEPT + tool_output
        else:
            delivered = _PREFIX_FILTERED + pruned_code

        attempt = CompressionAttempt(
            time_ms=elapsed_ms,
            usage=_usage_payload(
                prompt_tokens=origin_tokens,
                completion_tokens=left_tokens,
            ),
            extras={
                "model": self.name,
                "score": float(body.get("score", 0.0)),
                "kept_frags": list(body.get("kept_frags") or []),
                "origin_token_cnt": origin_tokens,
                "left_token_cnt": left_tokens,
                "model_input_token_cnt": int(body.get("model_input_token_cnt", 0)),
            },
        )
        return delivered, attempt
