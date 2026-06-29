"""Tests for the length reward module."""

import pytest

from src.reward.length_reward import compute_length_reward


class TestComputeLengthReward:
    """Tests for compute_length_reward."""

    def test_all_valid_same_length(self) -> None:
        """All valid completions with equal length collapse to zero reward."""
        rewards = compute_length_reward(
            compressed_outputs=["short", "short", "short"],
            action_rewards=[0.9, 0.9, 0.9],
            theta=0.8,
        )
        assert rewards == [0.0, 0.0, 0.0]

    def test_all_valid_different_lengths(self) -> None:
        """Shortest gets +0.5, longest gets -0.5, linear in between."""
        rewards = compute_length_reward(
            compressed_outputs=["short", "medium_len", "very_long_output"],
            action_rewards=[0.9, 0.9, 0.9],
            theta=0.8,
        )
        # lengths 5, 10, 16 → span 11
        assert rewards[0] == pytest.approx(0.5, rel=0.01)
        assert rewards[1] == pytest.approx(0.5 - 5 / 11, rel=0.01)
        assert rewards[2] == pytest.approx(-0.5, rel=0.01)

    def test_some_invalid(self) -> None:
        """Invalid completions (action_reward < theta) get 0.0 regardless of length."""
        rewards = compute_length_reward(
            compressed_outputs=["short", "longer", "longest"],
            action_rewards=[0.9, 0.5, 0.85],
            theta=0.8,
        )
        # only indices 0, 2 contribute to l_min/l_max: 5 and 7
        assert rewards[0] == pytest.approx(0.5, rel=0.01)
        assert rewards[1] == 0.0
        assert rewards[2] == pytest.approx(-0.5, rel=0.01)

    def test_all_invalid(self) -> None:
        """When no completion passes theta, all rewards are zero."""
        rewards = compute_length_reward(
            compressed_outputs=["short", "medium"],
            action_rewards=[0.3, 0.4],
            theta=0.8,
        )
        assert rewards == [0.0, 0.0]

    def test_empty_outputs(self) -> None:
        """Empty input returns empty list."""
        assert compute_length_reward([], [], theta=0.8) == []

    def test_single_valid_output(self) -> None:
        """A single valid completion has no peer to compare against → 0.0."""
        rewards = compute_length_reward(
            compressed_outputs=["output"],
            action_rewards=[0.9],
            theta=0.8,
        )
        assert rewards == [0.0]
