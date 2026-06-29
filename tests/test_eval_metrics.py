"""Tests for evaluation metrics extraction."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from src.eval.metrics import (
    AggregatedMetrics,
    MetricsExtractor,
    TrajectoryMetrics,
    bootstrap_mean_ci,
    compute_cost_stats,
    compute_pairwise_cost_significance,
    compute_pairwise_significance,
    compute_pass_at_1_stats,
    compute_run_significance,
    wilson_score_interval,
)


class TestTrajectoryMetrics:
    """Tests for TrajectoryMetrics dataclass."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should contain all metric fields."""
        metrics = TrajectoryMetrics(
            instance_id="test-1",
            api_calls=10,
            exit_status="success",
            has_submission=True,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            compression_time_ms=100.0,
            compressed_steps=5,
        )

        result = metrics.to_dict()

        assert result["instance_id"] == "test-1"
        assert result["api_calls"] == 10
        assert result["exit_status"] == "success"
        assert result["has_submission"] is True
        assert result["total_prompt_tokens"] == 1000
        assert result["total_completion_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["compression_time_ms"] == 100.0
        assert result["compressed_steps"] == 5


class TestMetricsExtractor:
    """Tests for MetricsExtractor class."""

    @pytest.fixture
    def sample_trajectory(self) -> dict:
        """Create a sample trajectory for testing."""
        return {
            "info": {
                "instance_id": "test-repo__test-123",
                "exit_status": "submitted",
                "submission": "Fixed the bug",
                "model_stats": {
                    "api_calls": 25,
                    "instance_cost": 0.0,
                },
            },
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Fix this bug"},
                {
                    "role": "assistant",
                    "content": "Let me check the file",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 1000,
                                "completion_tokens": 100,
                            }
                        }
                    },
                },
                {"role": "tool", "content": "File content here"},
                {
                    "role": "assistant",
                    "content": "Now I'll fix it",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 2000,
                                "completion_tokens": 150,
                            }
                        }
                    },
                },
            ],
        }

    def test_extract_basic_metrics(self, sample_trajectory: dict) -> None:
        """Test extraction of basic metrics."""
        metrics = MetricsExtractor.extract(sample_trajectory)

        assert metrics.instance_id == "test-repo__test-123"
        assert metrics.api_calls == 25
        assert metrics.exit_status == "submitted"
        assert metrics.has_submission is True

    def test_extract_token_metrics(self, sample_trajectory: dict) -> None:
        """Test extraction of token metrics."""
        metrics = MetricsExtractor.extract(sample_trajectory)

        # Should sum prompt_tokens from all assistant messages
        assert metrics.total_prompt_tokens == 3000  # 1000 + 2000
        assert metrics.total_completion_tokens == 250  # 100 + 150
        assert metrics.total_tokens == 3250

    def test_extract_no_submission(self, sample_trajectory: dict) -> None:
        """Test extraction when there's no submission."""
        sample_trajectory["info"]["submission"] = None

        metrics = MetricsExtractor.extract(sample_trajectory)

        assert metrics.has_submission is False

    def test_extract_empty_submission(self, sample_trajectory: dict) -> None:
        """Empty-string submissions should count as missing."""
        sample_trajectory["info"]["submission"] = ""

        metrics = MetricsExtractor.extract(sample_trajectory)

        assert metrics.has_submission is False

    def test_extract_with_compression_stats(self, sample_trajectory: dict) -> None:
        """Test extraction of compression stats."""
        sample_trajectory["info"]["compression_stats"] = {
            "total_time_ms": 500.0,
            "compressed_steps": 10,
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "cached_tokens": 8,
            "input_cost_usd": 0.11,
            "cache_read_cost_usd": 0.02,
            "cache_creation_cost_usd": 0.0,
            "output_cost_usd": 0.03,
            "total_cost_usd": 0.16,
        }

        metrics = MetricsExtractor.extract(sample_trajectory)

        assert metrics.compression_time_ms == 500.0
        assert metrics.compressed_steps == 10
        assert metrics.compression_prompt_tokens == 120
        assert metrics.compression_completion_tokens == 30
        assert metrics.compression_total_tokens == 150
        assert metrics.compression_cached_tokens == 8
        assert metrics.compression_total_cost_usd == 0.16

    def test_aggregate_empty_list(self) -> None:
        """Test aggregation with empty list."""
        result = MetricsExtractor.aggregate([], method="test")

        assert result.total_instances == 0
        assert result.avg_prompt_tokens == 0.0
        assert result.total_compressed_steps == 0

    def test_aggregate_multiple_metrics(self) -> None:
        """Test aggregation of multiple metrics."""
        metrics_list = [
            TrajectoryMetrics(
                instance_id="test-1",
                api_calls=10,
                exit_status="success",
                has_submission=True,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_tokens=1500,
                compression_time_ms=100.0,
                compressed_steps=2,
            ),
            TrajectoryMetrics(
                instance_id="test-2",
                api_calls=20,
                exit_status="success",
                has_submission=True,
                total_prompt_tokens=2000,
                total_completion_tokens=1000,
                total_tokens=3000,
                compression_time_ms=200.0,
                compressed_steps=4,
            ),
        ]

        result = MetricsExtractor.aggregate(metrics_list, method="test")

        assert result.total_instances == 2
        assert result.successful_submissions == 2
        assert result.total_prompt_tokens == 3000
        assert result.total_completion_tokens == 1500
        assert result.avg_prompt_tokens == 1500.0  # (1000 + 2000) / 2
        assert result.total_api_calls == 30  # 10 + 20
        assert result.avg_api_calls == 15.0
        assert result.avg_compression_time_ms == 150.0  # (100 + 200) / 2
        assert result.total_compressed_steps == 6
        assert result.avg_compressed_steps == 3.0

    def test_aggregate_compression_usage_and_cost(self) -> None:
        """Compression usage and cost should aggregate separately from main model cost."""
        metrics_list = [
            TrajectoryMetrics(
                instance_id="test-1",
                api_calls=10,
                exit_status="success",
                has_submission=True,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_tokens=1500,
                compression_prompt_tokens=40,
                compression_completion_tokens=10,
                compression_total_tokens=50,
                compression_cached_tokens=5,
                compression_input_cost_usd=0.11,
                compression_output_cost_usd=0.02,
                compression_total_cost_usd=0.13,
            ),
            TrajectoryMetrics(
                instance_id="test-2",
                api_calls=20,
                exit_status="success",
                has_submission=True,
                total_prompt_tokens=2000,
                total_completion_tokens=1000,
                total_tokens=3000,
                compression_prompt_tokens=60,
                compression_completion_tokens=15,
                compression_total_tokens=75,
                compression_cached_tokens=7,
                compression_input_cost_usd=0.07,
                compression_output_cost_usd=0.01,
                compression_total_cost_usd=0.08,
            ),
        ]

        result = MetricsExtractor.aggregate(metrics_list, method="test")

        assert result.total_compression_prompt_tokens == 100
        assert result.avg_compression_prompt_tokens == 50.0
        assert result.total_compression_completion_tokens == 25
        assert result.total_compression_tokens == 125
        assert result.total_compression_cached_tokens == 12
        assert result.total_compression_cost_usd == pytest.approx(0.21)
        assert result.avg_compression_cost_usd == pytest.approx(0.105)

    def test_aggregate_with_failures(self) -> None:
        """Test aggregation with some failed instances."""
        metrics_list = [
            TrajectoryMetrics(
                instance_id="test-1",
                api_calls=10,
                exit_status="success",
                has_submission=True,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_tokens=1500,
            ),
            TrajectoryMetrics(
                instance_id="test-2",
                api_calls=20,
                exit_status="error",
                has_submission=False,
                total_prompt_tokens=2000,
                total_completion_tokens=1000,
                total_tokens=3000,
            ),
        ]

        result = MetricsExtractor.aggregate(metrics_list, method="test")

        assert result.total_instances == 2
        assert result.successful_submissions == 1
        assert result.failed_submissions == 1

    def test_aggregate_with_dropped_failures(self) -> None:
        """Dropped failed runs should count as failures but not affect averages."""
        metrics_list = [
            TrajectoryMetrics(
                instance_id="test-1",
                api_calls=10,
                exit_status="success",
                has_submission=True,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_tokens=1500,
            )
        ]

        result = MetricsExtractor.aggregate(
            metrics_list,
            method="test",
            total_instances=3,
        )

        assert result.total_instances == 3
        assert result.successful_submissions == 1
        assert result.failed_submissions == 2
        assert result.avg_prompt_tokens == 1000.0
        assert result.avg_api_calls == 10.0


class TestAggregatedMetrics:
    """Tests for AggregatedMetrics dataclass."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should contain all aggregated fields."""
        metrics = AggregatedMetrics(
            method="test_method",
            total_instances=10,
            successful_submissions=8,
            failed_submissions=2,
            avg_prompt_tokens=1500.0,
            avg_completion_tokens=500.0,
            avg_total_tokens=2000.0,
            total_prompt_tokens=15000,
            total_completion_tokens=5000,
            avg_api_calls=15.0,
            total_api_calls=150,
            avg_compression_time_ms=100.0,
            total_compression_time_ms=1000.0,
            avg_compressed_steps=2.5,
            total_compressed_steps=25,
            trajectory_generation_time_seconds=3600.0,
            timestamp="2024-01-01T00:00:00",
        )

        result = metrics.to_dict()

        assert result["method"] == "test_method"
        assert result["total_instances"] == 10
        assert result["successful_submissions"] == 8
        assert result["submission_success_rate"] == 0.8
        assert result["avg_prompt_tokens"] == 1500.0
        assert result["avg_api_calls"] == 15.0
        assert result["avg_compressed_steps"] == 2.5
        assert result["total_compressed_steps"] == 25
        assert result["trajectory_generation_time_seconds"] == 3600.0


class TestWilsonScoreInterval:
    """Tests for wilson_score_interval."""

    def test_matches_statsmodels_proportion_confint(self) -> None:
        """Wrapper output must equal statsmodels.proportion_confint(method='wilson')."""
        low, high = wilson_score_interval(107, 200, confidence=0.95)
        ref_low, ref_high = proportion_confint(107, 200, alpha=0.05, method="wilson")
        assert low == pytest.approx(float(ref_low))
        assert high == pytest.approx(float(ref_high))

    def test_known_value_pass_at_one(self) -> None:
        """107/200 should give the documented Wilson 95% CI bounds."""
        low, high = wilson_score_interval(107, 200, confidence=0.95)
        assert low == pytest.approx(0.4658664687236316, abs=1e-9)
        assert high == pytest.approx(0.6028143584299599, abs=1e-9)

    def test_boundary_zero_successes(self) -> None:
        """Zero successes must give a non-negative lower bound."""
        low, high = wilson_score_interval(0, 50, confidence=0.95)
        assert low >= 0.0
        assert high > 0.0

    def test_boundary_all_successes(self) -> None:
        """All successes must give upper bound at 1.0."""
        low, high = wilson_score_interval(50, 50, confidence=0.95)
        assert high == pytest.approx(1.0, abs=1e-9)
        assert low < 1.0

    def test_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_interval(0, 0)

    def test_rejects_successes_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_interval(11, 10)

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError):
            wilson_score_interval(5, 10, confidence=1.5)


class TestPassAt1Stats:
    """Tests for compute_pass_at_1_stats."""

    def test_intersects_with_universe(self) -> None:
        """Resolved IDs outside the universe are ignored."""
        stats = compute_pass_at_1_stats(
            resolved_ids={"a", "b", "c", "z"},
            instance_ids={"a", "b", "c", "d"},
            method="m",
            confidence=0.95,
        )
        assert stats.n == 4
        assert stats.successes == 3  # "z" not in universe
        assert stats.pass_at_1 == pytest.approx(0.75)
        assert 0.0 <= stats.ci_low <= stats.pass_at_1 <= stats.ci_high <= 1.0


class TestPairwiseSignificance:
    """Tests for compute_pairwise_significance (paired McNemar)."""

    def test_contingency_table_counts(self) -> None:
        """Counts must split into both / only_a / only_b / neither correctly."""
        universe = {"i1", "i2", "i3", "i4", "i5"}
        a = {"i1", "i2", "i3"}
        b = {"i2", "i4"}
        result = compute_pairwise_significance("A", a, "B", b, universe)
        assert result.both_pass == 1  # i2
        assert result.only_a == 2  # i1, i3
        assert result.only_b == 1  # i4
        assert result.neither == 1  # i5
        assert result.n == 5
        assert result.delta == pytest.approx((3 - 2) / 5)

    def test_p_value_matches_statsmodels_exact_mcnemar(self) -> None:
        """The p-value must equal statsmodels.mcnemar with exact=True."""
        # b=17 only CoACT, c=29 only other -> discordant tail < 0.5, two-sided ~= 0.104
        universe = {f"i{i}" for i in range(200)}
        a = {f"i{i}" for i in range(107)}
        b = {f"i{i}" for i in range(90)} | {f"i{i}" for i in range(107, 136)}
        result = compute_pairwise_significance("CoACT", a, "other", b, universe)
        assert result.both_pass == 90
        assert result.only_a == 17
        assert result.only_b == 29
        ref = mcnemar([[90, 17], [29, 71]], exact=True).pvalue
        assert result.p_value == pytest.approx(float(ref), abs=1e-12)

    def test_no_discordant_pairs_returns_p_one(self) -> None:
        """Identical methods produce p-value 1.0."""
        universe = {"a", "b", "c"}
        result = compute_pairwise_significance("X", {"a", "b"}, "Y", {"a", "b"}, universe)
        assert result.only_a == 0 and result.only_b == 0
        assert result.p_value == pytest.approx(1.0)


class TestComputeRunSignificance:
    """End-to-end test for compute_run_significance with synthetic report dirs."""

    def _write_report(self, path: Path, resolved: list[str], submitted: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "resolved_ids": resolved,
                    "submitted_ids": submitted,
                    "total_instances": len(submitted),
                }
            )
        )

    def test_reads_reports_and_runs_pairwise_tests(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        instances = [f"i{i}" for i in range(10)]
        self._write_report(
            run / "results" / "CoACT" / "swebench" / "report.json",
            resolved=instances[:5],
            submitted=instances,
        )
        self._write_report(
            run / "results" / "vanilla" / "swebench" / "report.json",
            resolved=instances[3:8],
            submitted=instances,
        )

        out = compute_run_significance(run, target_method="CoACT", confidence=0.95)

        assert out["shared_instances"] == 10
        assert {entry["method"] for entry in out["per_method"]} == {"CoACT", "vanilla"}
        assert len(out["pairwise"]) == 1
        comparison = out["pairwise"][0]
        assert comparison["method_a"] == "CoACT"
        assert comparison["method_b"] == "vanilla"
        assert math.isfinite(comparison["p_value"])
        assert 0.0 <= comparison["p_value"] <= 1.0

    def test_missing_target_method_raises(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        self._write_report(
            run / "results" / "baseline" / "swebench" / "report.json",
            resolved=["a"],
            submitted=["a", "b"],
        )
        with pytest.raises(ValueError):
            compute_run_significance(run, target_method="CoACT")


class TestBootstrapMeanCi:
    """Tests for bootstrap_mean_ci."""

    def test_deterministic_with_seed(self) -> None:
        """Same seed must yield identical CI bounds across calls."""
        rng = np.random.default_rng(123)
        values = rng.normal(loc=1.5, scale=0.5, size=200).tolist()
        first = bootstrap_mean_ci(values, rng_seed=7)
        second = bootstrap_mean_ci(values, rng_seed=7)
        assert first == second

    def test_ci_contains_sample_mean(self) -> None:
        """Bootstrap mean CI should bracket the sample mean."""
        rng = np.random.default_rng(0)
        values = rng.normal(loc=2.0, scale=0.3, size=300).tolist()
        low, high = bootstrap_mean_ci(values, rng_seed=0)
        assert low < float(np.mean(values)) < high

    def test_constant_sample_collapses_to_point(self) -> None:
        """A constant sample returns a zero-width CI at that value."""
        low, high = bootstrap_mean_ci([3.0, 3.0, 3.0, 3.0], rng_seed=0)
        assert low == pytest.approx(3.0)
        assert high == pytest.approx(3.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_mean_ci([], rng_seed=0)


class TestComputeCostStats:
    """Tests for compute_cost_stats."""

    def test_basic_summary_fields(self) -> None:
        costs = {"i1": 1.0, "i2": 2.0, "i3": 3.0, "i4": 4.0, "i5": 5.0}
        stats = compute_cost_stats("m", costs, confidence=0.95, rng_seed=0)
        assert stats.n == 5
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.total == pytest.approx(15.0)
        assert stats.std == pytest.approx(float(np.std([1, 2, 3, 4, 5], ddof=1)))
        assert stats.ci_low < stats.mean < stats.ci_high


class TestPairwiseCostSignificance:
    """Tests for compute_pairwise_cost_significance."""

    def test_pairs_on_intersection_and_wilcoxon_matches_scipy(self) -> None:
        """Diffs are taken on shared keys; p-value must equal scipy.wilcoxon."""
        rng = np.random.default_rng(7)
        ids = [f"i{i}" for i in range(40)]
        cost_a = {i: float(v) for i, v in zip(ids, rng.normal(1.0, 0.2, len(ids)), strict=True)}
        cost_b = {i: float(v) for i, v in zip(ids, rng.normal(1.4, 0.2, len(ids)), strict=True)}
        result = compute_pairwise_cost_significance("A", cost_a, "B", cost_b, rng_seed=0)

        diffs = np.array([cost_a[i] - cost_b[i] for i in ids])
        ref_p = float(scipy_stats.wilcoxon(diffs, zero_method="wilcox").pvalue)
        assert result.n == 40
        assert result.mean_diff == pytest.approx(float(diffs.mean()))
        assert result.wilcoxon_pvalue == pytest.approx(ref_p, abs=1e-12)
        assert result.diff_ci_low < result.mean_diff < result.diff_ci_high

    def test_identical_costs_returns_p_one(self) -> None:
        """All-zero diffs trigger the degenerate-case fallback."""
        costs = {f"i{i}": float(i) for i in range(5)}
        result = compute_pairwise_cost_significance("A", costs, "B", costs, rng_seed=0)
        assert result.mean_diff == pytest.approx(0.0)
        assert result.wilcoxon_pvalue == pytest.approx(1.0)

    def test_no_shared_instances_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_pairwise_cost_significance(
                "A", {"a": 1.0}, "B", {"b": 1.0}, rng_seed=0
            )
