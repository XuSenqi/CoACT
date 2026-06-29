"""Reward calculation module for SFT training.

This module provides reward functions for evaluating:
- Action similarity (bash command similarity)
- Length reward (compression efficiency)
"""

from src.reward.bash_similarity import (
    BashParseError,
    CommandSemantics,
    compute_command_similarity,
    compute_semantic_similarity,
    extract_semantics,
    parse_bash_command,
)
from src.reward.length_reward import compute_length_reward

__all__ = [
    # Bash similarity
    "BashParseError",
    "CommandSemantics",
    "compute_command_similarity",
    "compute_semantic_similarity",
    "extract_semantics",
    "parse_bash_command",
    # Length reward
    "compute_length_reward",
]
