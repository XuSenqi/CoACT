"""Compression filter for determining when to bypass context compression.

This module provides logic for deciding whether tool output should skip
compression based on:
- Short output that fits within token limits
- Unparseable bash commands
"""

import logging
from collections.abc import Sequence
from enum import Enum
from typing import Any

from src.reward.bash_similarity import BashParseError, parse_bash_command

logger = logging.getLogger(__name__)


class CompressionFilterReason(str, Enum):
    """Reasons for filtering a step out of compression."""

    SHORT_OUTPUT = "short_output"
    UNPARSEABLE_COMMAND = "unparseable_command"


class CompressionFilter:
    """Determine whether a tool output should bypass compression."""

    def __init__(
        self,
        skip_compression_max_tokens: int,
        tokenizer: Any,
    ) -> None:
        """Initialize the compression filter.

        Args:
            skip_compression_max_tokens: Tool outputs with token count at or below
                this threshold are skipped without compression.
            tokenizer: Tokenizer used for counting tokens.
        """
        self.skip_compression_max_tokens = skip_compression_max_tokens
        self._tokenizer = tokenizer

    @classmethod
    def build(cls, skip_compression_max_tokens: int) -> "CompressionFilter":
        """Build a filter using tiktoken for token counting.

        Args:
            skip_compression_max_tokens: Token threshold below which compression is skipped.

        Returns:
            CompressionFilter initialized with a tiktoken tokenizer.

        Raises:
            ImportError: If `tiktoken` is not installed.
        """
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for token-based compression filtering"
            ) from exc

        return cls(skip_compression_max_tokens, tokenizer)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens in the text.
        """
        return len(self._tokenizer.encode(text))

    def get_filter_reason(
        self,
        tool_calls: Sequence[str],
        tool_output_str: str,
        step_index: int | None = None,
    ) -> CompressionFilterReason | None:
        """Determine whether a tool output should bypass compression.

        Args:
            tool_calls: Sequence of bash commands executed.
            tool_output_str: Combined tool output string.
            step_index: Optional step index for logging.

        Returns:
            CompressionFilterReason if compression should be skipped, else `None`.
        """
        for command in tool_calls:
            try:
                parse_bash_command(command)
            except BashParseError:
                if step_index is None:
                    logger.warning(
                        f"Filtering tool output due to unparseable bash command: {command!r}"
                    )
                else:
                    logger.warning(
                        f"Filtering step {step_index} due to unparseable bash command: "
                        f"{command!r}"
                    )
                return CompressionFilterReason.UNPARSEABLE_COMMAND

        if self.count_tokens(tool_output_str) <= self.skip_compression_max_tokens:
            return CompressionFilterReason.SHORT_OUTPUT

        return None
