"""Sliding-window compression baseline.

This strategy doesn't compress individual tool outputs — its real work
happens in ``CompressingAgent.execute_actions`` where it redacts older
tool messages once they fall outside the window.
"""

from typing import Any

from .base import CompressionAttempt, CompressionStrategy


class SlidingWindow(CompressionStrategy):
    """Sliding window - keep only the most recent N tool outputs.

    This strategy doesn't compress individual outputs, but instead
    removes older tool outputs when the count exceeds window_size,
    keeping only the most recent ones.
    """

    name = "sliding_window"
    removed_content = "[Tool output removed by sliding window]"

    def __init__(self, window_size: int = 10):
        """Initialize sliding window strategy.

        Args:
            window_size: Maximum number of tool outputs to keep.
        """
        self.window_size = window_size

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        """For sliding window, we don't modify individual outputs."""
        return tool_output, None

    def get_message_indices_to_remove(self, messages: list[dict[str, Any]]) -> list[int]:
        """Get absolute message indices that should be redacted.

        Args:
            messages: Full agent message history.

        Returns:
            Absolute indices into ``messages`` for tool messages that fall
            outside the current sliding window and have not been redacted yet.
        """
        tool_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "tool"]
        num_to_remove = len(tool_indices) - self.window_size
        if num_to_remove <= 0:
            return []

        indices_to_remove: list[int] = []
        for msg_idx in tool_indices[:num_to_remove]:
            if messages[msg_idx].get("content") != self.removed_content:
                indices_to_remove.append(msg_idx)

        return indices_to_remove
