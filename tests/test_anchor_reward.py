"""Unit tests for the shared agent-natural anchor reward scorer."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from src.compression.anchor_reward import (
    AnchorAggregator,
    aggregate_similarities,
    sample_anchor_set,
    score_candidate_action,
)


class TestAnchorAggregator:
    def test_aggregate_mean_max_top3_count(self) -> None:
        sims = [0.2, 0.9, 0.5, 0.8]
        assert aggregate_similarities(sims, AnchorAggregator("mean")) == pytest.approx(0.6)
        assert aggregate_similarities(sims, AnchorAggregator("max")) == pytest.approx(0.9)
        assert aggregate_similarities(sims, AnchorAggregator("top3")) == pytest.approx(
            (0.9 + 0.8 + 0.5) / 3
        )
        assert aggregate_similarities(
            sims, AnchorAggregator("count", count_threshold=0.5)
        ) == pytest.approx(0.75)

    def test_empty_sims_is_zero(self) -> None:
        assert aggregate_similarities([], AnchorAggregator("mean")) == 0.0
        assert aggregate_similarities([], AnchorAggregator("top3")) == 0.0

    def test_top3_with_fewer_than_three(self) -> None:
        assert aggregate_similarities([0.4, 0.6], AnchorAggregator("top3")) == pytest.approx(0.5)

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="anchor aggregator"):
            AnchorAggregator(kind="bogus")

    def test_count_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="count_threshold"):
            AnchorAggregator(kind="count", count_threshold=1.5)

    def test_from_config_reads_fields(self) -> None:
        class _Cfg:
            anchor_aggregator = "mean"
            anchor_count_threshold = 0.3

        aggregator = AnchorAggregator.from_config(_Cfg())
        assert aggregator.kind == "mean"
        assert aggregator.count_threshold == pytest.approx(0.3)


class TestScoreCandidateAction:
    def test_wires_similarity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sim_map = {("cmd", "a"): 0.2, ("cmd", "b"): 0.9, ("cmd", "c"): 0.7}
        monkeypatch.setattr(
            "src.compression.anchor_reward.compute_command_similarity",
            lambda x, y: sim_map[(x, y)],
        )
        score = score_candidate_action("cmd", ["a", "b", "c"], AnchorAggregator("mean"))
        assert score == pytest.approx((0.2 + 0.9 + 0.7) / 3)

    def test_empty_anchor_set_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.compression.anchor_reward.compute_command_similarity",
            lambda x, y: 1.0,
        )
        assert score_candidate_action("cmd", [], AnchorAggregator("mean")) == 0.0


class _FakeRunner:
    """Records ``continue_from_step`` calls and returns scripted outputs."""

    def __init__(self, outputs: list[str | None]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def continue_from_step(self, **kwargs: Any) -> str | None:
        with self._lock:
            index = len(self.calls)
            self.calls.append(kwargs)
        return self._outputs[index]


class TestSampleAnchorSet:
    def test_calls_runner_k_times_and_drops_blank(self) -> None:
        runner = _FakeRunner(["cmd a", "   ", None, "cmd b", "cmd c"])
        prefix: list[dict[str, Any]] = [{"role": "user", "content": "x"}]

        result = asyncio.run(
            sample_anchor_set(
                runner,  # type: ignore[arg-type]
                prefix_messages=prefix,
                step_index=7,
                num_samples=5,
                max_workers=2,
            )
        )

        assert len(runner.calls) == 5
        for call in runner.calls:
            assert call["compressed_output"] == ""
            assert call["use_original_output"] is True
            assert call["step_index"] == 7
            assert call["trajectory_messages"] is prefix
        # Blank and None completions are dropped; order is not guaranteed.
        assert sorted(result) == ["cmd a", "cmd b", "cmd c"]

    def test_all_blank_yields_empty(self) -> None:
        runner = _FakeRunner([None, "", "   "])
        result = asyncio.run(
            sample_anchor_set(
                runner,  # type: ignore[arg-type]
                prefix_messages=[{"role": "user", "content": "x"}],
                step_index=0,
                num_samples=3,
                max_workers=4,
            )
        )
        assert result == []
