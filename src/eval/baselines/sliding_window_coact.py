"""Hybrid Sliding Window + CoACT compression strategy."""

from typing import Any

from src.config.config import Config

from .base import CompressionAttempt, CompressionStrategy
from .coact import CoACTCompression
from .sliding_window import SlidingWindow


class SlidingWindowCoACTCompression(CompressionStrategy):
    """Apply CoACT observation compression before sliding-window redaction.

    The paper's RQ2 setup first compresses each raw observation before it enters
    the trajectory, then applies the trajectory-level compressor to historical
    content. This wrapper keeps those two pieces explicit while presenting a
    single strategy to the evaluation runner.
    """

    name = "sliding_window_CoACT"
    consumes_cfq = True
    announces_compression = False

    def __init__(
        self,
        observation_strategy: CompressionStrategy,
        sliding_window: SlidingWindow,
    ) -> None:
        """Initialize the hybrid strategy.

        Args:
            observation_strategy: CoACT-compatible observation compressor.
            sliding_window: Sliding-window trajectory compressor.
        """
        self.observation_strategy = observation_strategy
        self.sliding_window = sliding_window

    @classmethod
    def from_config(cls, config: Config) -> "SlidingWindowCoACTCompression":
        """Create the RQ2 hybrid strategy from configuration.

        Args:
            config: The unified configuration.

        Returns:
            SlidingWindowCoACTCompression configured with ``evaluation.coact`` and
            ``evaluation.sliding_window_size``.
        """
        return cls(
            observation_strategy=CoACTCompression.from_config(config),
            sliding_window=SlidingWindow(window_size=config.evaluation.sliding_window_size),
        )

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """Compress one observation using the CoACT observation compressor.

        Args:
            tool_output: The original tool output to compress.
            context: Runtime compression context.

        Returns:
            The delegated CoACT compression result and attempt metadata.
        """
        return self.observation_strategy.compress(tool_output, context)

    def get_message_indices_to_remove(self, messages: list[dict[str, Any]]) -> list[int]:
        """Return historical tool-message indices to redact.

        Args:
            messages: Full agent message history.

        Returns:
            Absolute message indices selected by the sliding-window component.
        """
        return self.sliding_window.get_message_indices_to_remove(messages)
