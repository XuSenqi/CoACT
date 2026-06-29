"""Vanilla uncompressed baseline."""

from typing import Any

from .base import CompressionAttempt, CompressionStrategy


class VanillaCompression(CompressionStrategy):
    """Vanilla baseline - return the original output unchanged."""

    name = "vanilla"

    def compress(
        self,
        tool_output: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionAttempt | None]:
        return tool_output, None
