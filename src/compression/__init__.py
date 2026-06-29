"""Compression helpers shared by training and inference."""

from src.compression.compression_filter import CompressionFilter, CompressionFilterReason
from src.compression.output_parser import (
    CompressionOutputResolution,
    interpret_compression_response,
)
from src.compression.prompt_context import (
    CompressionPromptContext,
    build_runtime_compression_context,
    build_step_compression_context,
    has_context_focus_question,
    number_prompt_lines,
)
from src.compression.rollout_selection import (
    create_prompt_completion_examples,
    select_rollout_samples,
    split_rollout_records_by_group,
)

__all__ = [
    "CompressionPromptContext",
    "CompressionOutputResolution",
    "CompressionFilter",
    "CompressionFilterReason",
    "build_runtime_compression_context",
    "build_step_compression_context",
    "number_prompt_lines",
    "has_context_focus_question",
    "interpret_compression_response",
    "create_prompt_completion_examples",
    "select_rollout_samples",
    "split_rollout_records_by_group",
]
