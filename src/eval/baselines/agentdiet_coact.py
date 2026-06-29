"""Hybrid AgentDiet + CoACT compression strategy for trajectory compression."""

from typing import Any

from src.config.config import Config

from .agentdiet import AgentDietCompression, AgentDietReductionResult
from .base import CompressionAttempt, CompressionStrategy
from .coact import CoACTCompression


class AgentDietCoACTCompression(CompressionStrategy):
    """Apply CoACT observation compression before AgentDiet history reduction."""

    name = "agentdiet_CoACT"
    consumes_cfq = True
    announces_compression = False

    def __init__(
        self,
        observation_strategy: CompressionStrategy,
        agentdiet_strategy: AgentDietCompression,
    ) -> None:
        """Initialize the hybrid strategy.

        Args:
            observation_strategy: CoACT-compatible observation compressor.
            agentdiet_strategy: AgentDiet trajectory compressor.
        """
        self.observation_strategy = observation_strategy
        self.agentdiet_strategy = agentdiet_strategy

    @classmethod
    def from_config(cls, config: Config) -> "AgentDietCoACTCompression":
        """Create the AgentDiet + CoACT strategy from configuration.

        Args:
            config: The unified configuration.

        Returns:
            AgentDietCoACTCompression configured with ``evaluation.coact`` and
            ``evaluation.agentdiet``.
        """
        return cls(
            observation_strategy=CoACTCompression.from_config(config),
            agentdiet_strategy=AgentDietCompression.from_config(config),
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

    def reduce_message_in_history(
        self,
        messages: list[dict[str, Any]],
    ) -> AgentDietReductionResult:
        """Reduce historical trajectory content using AgentDiet.

        Args:
            messages: Current agent message history.

        Returns:
            The delegated AgentDiet reduction result.
        """
        return self.agentdiet_strategy.reduce_message_in_history(messages)
