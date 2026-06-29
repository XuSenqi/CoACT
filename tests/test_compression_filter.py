"""Tests for `CompressionFilter` in `src/compression/compression_filter.py`."""

from unittest.mock import patch

import pytest

from src.compression.compression_filter import (
    CompressionFilter,
    CompressionFilterReason,
)
from src.config.config import Config
from src.reward.bash_similarity import BashParseError


class _FakeTokenizer:
    """Fake tokenizer for testing token counting."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return a token count based on whitespace splitting."""
        del add_special_tokens
        return list(range(len(text.split())))


@pytest.fixture
def compression_filter() -> CompressionFilter:
    """Create a CompressionFilter using the config file."""
    config = Config.load("config/config.yaml")
    return CompressionFilter(config.sft.data_preparation.skip_compression_max_tokens, _FakeTokenizer())


class TestCompressionFilter:
    """Tests for CompressionFilter.get_filter_reason."""

    def test_get_filter_reason_for_short_output(
        self,
        compression_filter: CompressionFilter,
    ) -> None:
        """Test that short output is filtered due to token count."""
        reason = compression_filter.get_filter_reason(
            tool_calls=("grep TODO app.py",),
            tool_output_str="small output",
            step_index=1,
        )
        assert reason == CompressionFilterReason.SHORT_OUTPUT

    def test_get_filter_reason_returns_none_for_parseable_long_output(
        self,
        compression_filter: CompressionFilter,
    ) -> None:
        """Test that parseable long outputs are not filtered."""
        reason = compression_filter.get_filter_reason(
            tool_calls=("sed -n '1,40p' app.py",),
            tool_output_str="long enough output " * 200,
            step_index=2,
        )
        assert reason is None

    def test_get_filter_reason_for_unparseable_command(
        self,
        compression_filter: CompressionFilter,
    ) -> None:
        """Test that unparseable bash commands result in filtering."""
        with patch(
            "src.compression.compression_filter.parse_bash_command",
            side_effect=BashParseError("failed to parse"),
        ):
            reason = compression_filter.get_filter_reason(
                tool_calls=("python -c 'print(1)'",),
                tool_output_str="long enough output " * 200,
                step_index=3,
            )

        assert reason == CompressionFilterReason.UNPARSEABLE_COMMAND
