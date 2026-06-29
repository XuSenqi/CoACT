"""Metrics extraction from agent trajectories."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as _scipy_stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from src.utils.cost_tracker import get_cost_breakdown_from_usage, get_response_cost_breakdown


@dataclass
class TrajectoryMetrics:
    """Metrics extracted from a single trajectory."""

    instance_id: str

    # Basic stats
    api_calls: int  # Number of API calls
    exit_status: str
    has_submission: bool

    # Observed token stats from provider usage
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

    # Compression stats (if applicable)
    compression_time_ms: float = 0.0  # Time spent on compression
    compressed_steps: int = 0  # Number of assistant steps where compression was applied
    compression_prompt_tokens: int = 0
    compression_completion_tokens: int = 0
    compression_total_tokens: int = 0
    compression_cached_tokens: int = 0

    # Cost tracking (0.0 when pricing not configured)
    total_cached_tokens: int = 0            # Input tokens served from prefix cache
    input_cost_usd: float = 0.0             # Cost for non-cached input tokens
    cache_read_cost_usd: float = 0.0        # Cost for cache-hit input tokens
    cache_creation_cost_usd: float = 0.0    # Cost for cache-write tokens (Anthropic)
    output_cost_usd: float = 0.0            # Cost for output tokens
    total_cost_usd: float = 0.0             # Sum of agent + compression model costs

    # Compression-model cost tracking from runtime compression_stats
    compression_input_cost_usd: float = 0.0
    compression_cache_read_cost_usd: float = 0.0
    compression_cache_creation_cost_usd: float = 0.0
    compression_output_cost_usd: float = 0.0
    compression_total_cost_usd: float = 0.0

    # Timestamps
    collected_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "api_calls": self.api_calls,
            "exit_status": self.exit_status,
            "has_submission": self.has_submission,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "input_cost_usd": self.input_cost_usd,
            "cache_read_cost_usd": self.cache_read_cost_usd,
            "cache_creation_cost_usd": self.cache_creation_cost_usd,
            "output_cost_usd": self.output_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "compression_input_cost_usd": self.compression_input_cost_usd,
            "compression_cache_read_cost_usd": self.compression_cache_read_cost_usd,
            "compression_cache_creation_cost_usd": self.compression_cache_creation_cost_usd,
            "compression_output_cost_usd": self.compression_output_cost_usd,
            "compression_total_cost_usd": self.compression_total_cost_usd,
            "compression_time_ms": self.compression_time_ms,
            "compressed_steps": self.compressed_steps,
            "compression_prompt_tokens": self.compression_prompt_tokens,
            "compression_completion_tokens": self.compression_completion_tokens,
            "compression_total_tokens": self.compression_total_tokens,
            "compression_cached_tokens": self.compression_cached_tokens,
            "collected_at": self.collected_at,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple trajectories."""

    method: str
    total_instances: int
    successful_submissions: int  # Has submission
    failed_submissions: int

    # Token averages
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    total_prompt_tokens: int
    total_completion_tokens: int

    # API calls
    avg_api_calls: float
    total_api_calls: int

    # Compression stats
    avg_compression_time_ms: float
    total_compression_time_ms: float
    avg_compressed_steps: float = 0.0
    total_compressed_steps: int = 0
    total_compression_prompt_tokens: int = 0
    avg_compression_prompt_tokens: float = 0.0
    total_compression_completion_tokens: int = 0
    avg_compression_completion_tokens: float = 0.0
    total_compression_tokens: int = 0
    avg_compression_tokens: float = 0.0
    total_compression_cached_tokens: int = 0
    avg_compression_cached_tokens: float = 0.0

    # Cost tracking (0.0 when pricing not configured)
    total_input_cost_usd: float = 0.0
    avg_input_cost_usd: float = 0.0
    total_cache_read_cost_usd: float = 0.0
    avg_cache_read_cost_usd: float = 0.0
    total_cache_creation_cost_usd: float = 0.0
    avg_cache_creation_cost_usd: float = 0.0
    total_output_cost_usd: float = 0.0
    avg_output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0

    # Compression-model cost totals/averages
    total_compression_input_cost_usd: float = 0.0
    avg_compression_input_cost_usd: float = 0.0
    total_compression_cache_read_cost_usd: float = 0.0
    avg_compression_cache_read_cost_usd: float = 0.0
    total_compression_cache_creation_cost_usd: float = 0.0
    avg_compression_cache_creation_cost_usd: float = 0.0
    total_compression_output_cost_usd: float = 0.0
    avg_compression_output_cost_usd: float = 0.0
    total_compression_cost_usd: float = 0.0
    avg_compression_cost_usd: float = 0.0

    # Timing
    trajectory_generation_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "total_instances": self.total_instances,
            "successful_submissions": self.successful_submissions,
            "failed_submissions": self.failed_submissions,
            "submission_success_rate": self.successful_submissions / max(self.total_instances, 1),
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "avg_api_calls": self.avg_api_calls,
            "total_api_calls": self.total_api_calls,
            "avg_compression_time_ms": self.avg_compression_time_ms,
            "total_compression_time_ms": self.total_compression_time_ms,
            "avg_compressed_steps": self.avg_compressed_steps,
            "total_compressed_steps": self.total_compressed_steps,
            "total_compression_prompt_tokens": self.total_compression_prompt_tokens,
            "avg_compression_prompt_tokens": self.avg_compression_prompt_tokens,
            "total_compression_completion_tokens": self.total_compression_completion_tokens,
            "avg_compression_completion_tokens": self.avg_compression_completion_tokens,
            "total_compression_tokens": self.total_compression_tokens,
            "avg_compression_tokens": self.avg_compression_tokens,
            "total_compression_cached_tokens": self.total_compression_cached_tokens,
            "avg_compression_cached_tokens": self.avg_compression_cached_tokens,
            "total_input_cost_usd": self.total_input_cost_usd,
            "avg_input_cost_usd": self.avg_input_cost_usd,
            "total_cache_read_cost_usd": self.total_cache_read_cost_usd,
            "avg_cache_read_cost_usd": self.avg_cache_read_cost_usd,
            "total_cache_creation_cost_usd": self.total_cache_creation_cost_usd,
            "avg_cache_creation_cost_usd": self.avg_cache_creation_cost_usd,
            "total_output_cost_usd": self.total_output_cost_usd,
            "avg_output_cost_usd": self.avg_output_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_usd": self.avg_cost_usd,
            "total_compression_input_cost_usd": self.total_compression_input_cost_usd,
            "avg_compression_input_cost_usd": self.avg_compression_input_cost_usd,
            "total_compression_cache_read_cost_usd": self.total_compression_cache_read_cost_usd,
            "avg_compression_cache_read_cost_usd": self.avg_compression_cache_read_cost_usd,
            "total_compression_cache_creation_cost_usd": self.total_compression_cache_creation_cost_usd,
            "avg_compression_cache_creation_cost_usd": self.avg_compression_cache_creation_cost_usd,
            "total_compression_output_cost_usd": self.total_compression_output_cost_usd,
            "avg_compression_output_cost_usd": self.avg_compression_output_cost_usd,
            "total_compression_cost_usd": self.total_compression_cost_usd,
            "avg_compression_cost_usd": self.avg_compression_cost_usd,
            "trajectory_generation_time_seconds": self.trajectory_generation_time_seconds,
            "timestamp": self.timestamp,
        }


class MetricsExtractor:
    """Extract and aggregate metrics from trajectory files."""

    @staticmethod
    def has_valid_submission(submission: Any) -> bool:
        """Return whether a trajectory contains a non-empty submission."""
        if isinstance(submission, str):
            return bool(submission.strip())
        return submission is not None

    @staticmethod
    def extract(
        trajectory: dict[str, Any],
        model: str | None = None,
    ) -> TrajectoryMetrics:
        """Extract metrics from a trajectory dictionary.

        Args:
            trajectory: The trajectory data (loaded from JSON).
            model: LiteLLM model identifier used for the run. When provided,
                cost is computed via litellm.completion_cost() for each
                assistant message. Omit to skip cost tracking.

        Returns:
            TrajectoryMetrics with extracted values.
        """
        info = trajectory.get("info", {})
        messages = trajectory.get("messages", [])
        summary_metrics = trajectory.get("metrics", {})
        model_stats = info.get("model_stats", {})

        # Basic stats
        instance_id = info.get("instance_id", "unknown")
        api_calls = model_stats.get("api_calls", 0)
        exit_status = info.get("exit_status", "unknown")
        has_submission = MetricsExtractor.has_valid_submission(info.get("submission"))

        # Token stats - sum from assistant messages
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        input_cost_usd = 0.0
        cache_read_cost_usd = 0.0
        cache_creation_cost_usd = 0.0
        output_cost_usd = 0.0

        for msg in messages:
            if msg["role"] == "assistant" and "extra" in msg:
                extra = msg["extra"]
                response = extra["response"]
                usage = response["usage"]
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]

                prompt_details = usage.get("prompt_tokens_details")
                if isinstance(prompt_details, dict):
                    total_cached_tokens += prompt_details.get("cached_tokens", 0)

                if model is not None:
                    breakdown = get_response_cost_breakdown(response, model)
                    input_cost_usd += breakdown.input_usd
                    cache_read_cost_usd += breakdown.cache_read_usd
                    cache_creation_cost_usd += breakdown.cache_creation_usd
                    output_cost_usd += breakdown.output_usd

        if total_prompt_tokens == 0 and isinstance(summary_metrics, dict):
            total_prompt_tokens = int(summary_metrics.get("prompt_tokens", 0) or 0)
            total_completion_tokens = int(summary_metrics.get("completion_tokens", 0) or 0)
            total_cached_tokens = int(summary_metrics.get("cached_tokens", 0) or 0)

            if model is not None:
                breakdown = get_cost_breakdown_from_usage(
                    model=model,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    cached_tokens=total_cached_tokens,
                )
                input_cost_usd = breakdown.input_usd
                cache_read_cost_usd = breakdown.cache_read_usd
                cache_creation_cost_usd = breakdown.cache_creation_usd
                output_cost_usd = breakdown.output_usd

        agent_cost_usd = (
            input_cost_usd + cache_read_cost_usd + cache_creation_cost_usd + output_cost_usd
        )

        # Compression stats from extra data (if present)
        compression_time_ms = 0.0
        compressed_steps = 0
        compression_prompt_tokens = 0
        compression_completion_tokens = 0
        compression_total_tokens = 0
        compression_cached_tokens = 0
        compression_input_cost_usd = 0.0
        compression_cache_read_cost_usd = 0.0
        compression_cache_creation_cost_usd = 0.0
        compression_output_cost_usd = 0.0
        compression_total_cost_usd = 0.0
        comp_stats = info.get("compression_stats")
        if isinstance(comp_stats, dict):
            compression_time_ms = float(comp_stats.get("total_time_ms", 0.0) or 0.0)
            compressed_steps = int(comp_stats.get("compressed_steps", 0) or 0)
            compression_prompt_tokens = int(comp_stats.get("prompt_tokens", 0) or 0)
            compression_completion_tokens = int(comp_stats.get("completion_tokens", 0) or 0)
            compression_total_tokens = int(comp_stats.get("total_tokens", 0) or 0)
            compression_cached_tokens = int(comp_stats.get("cached_tokens", 0) or 0)
            compression_input_cost_usd = float(comp_stats.get("input_cost_usd", 0.0) or 0.0)
            compression_cache_read_cost_usd = float(
                comp_stats.get("cache_read_cost_usd", 0.0) or 0.0
            )
            compression_cache_creation_cost_usd = float(
                comp_stats.get("cache_creation_cost_usd", 0.0) or 0.0
            )
            compression_output_cost_usd = float(comp_stats.get("output_cost_usd", 0.0) or 0.0)
            compression_total_cost_usd = float(comp_stats.get("total_cost_usd", 0.0) or 0.0)

        total_cost_usd = agent_cost_usd + compression_total_cost_usd

        # Timestamp
        collection = info.get("collection")
        collected_at = collection.get("collected_at", "") if isinstance(collection, dict) else ""

        return TrajectoryMetrics(
            instance_id=instance_id,
            api_calls=api_calls,
            exit_status=exit_status,
            has_submission=has_submission,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            total_cached_tokens=total_cached_tokens,
            input_cost_usd=input_cost_usd,
            cache_read_cost_usd=cache_read_cost_usd,
            cache_creation_cost_usd=cache_creation_cost_usd,
            output_cost_usd=output_cost_usd,
            total_cost_usd=total_cost_usd,
            compression_input_cost_usd=compression_input_cost_usd,
            compression_cache_read_cost_usd=compression_cache_read_cost_usd,
            compression_cache_creation_cost_usd=compression_cache_creation_cost_usd,
            compression_output_cost_usd=compression_output_cost_usd,
            compression_total_cost_usd=compression_total_cost_usd,
            compression_time_ms=compression_time_ms,
            compressed_steps=compressed_steps,
            compression_prompt_tokens=compression_prompt_tokens,
            compression_completion_tokens=compression_completion_tokens,
            compression_total_tokens=compression_total_tokens,
            compression_cached_tokens=compression_cached_tokens,
            collected_at=collected_at,
        )

    @staticmethod
    def aggregate(
        metrics_list: list[TrajectoryMetrics],
        method: str,
        trajectory_generation_time_seconds: float = 0.0,
        total_instances: int | None = None,
    ) -> AggregatedMetrics:
        """Aggregate metrics from multiple trajectories.

        Args:
            metrics_list: List of TrajectoryMetrics to aggregate.
            method: Name of the compression method.
            trajectory_generation_time_seconds: Time spent generating trajectories.
            total_instances: Total number of requested instances, including dropped failures.

        Returns:
            AggregatedMetrics with computed averages and totals.
        """
        instance_count = total_instances if total_instances is not None else len(metrics_list)

        if not metrics_list:
            return AggregatedMetrics(
                method=method,
                total_instances=instance_count,
                successful_submissions=0,
                failed_submissions=instance_count,
                avg_prompt_tokens=0.0,
                avg_completion_tokens=0.0,
                avg_total_tokens=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                avg_api_calls=0.0,
                total_api_calls=0,
                avg_compression_time_ms=0.0,
                total_compression_time_ms=0.0,
                avg_compressed_steps=0.0,
                total_compressed_steps=0,
                total_compression_prompt_tokens=0,
                avg_compression_prompt_tokens=0.0,
                total_compression_completion_tokens=0,
                avg_compression_completion_tokens=0.0,
                total_compression_tokens=0,
                avg_compression_tokens=0.0,
                total_compression_cached_tokens=0,
                avg_compression_cached_tokens=0.0,
                total_input_cost_usd=0.0,
                avg_input_cost_usd=0.0,
                total_cache_read_cost_usd=0.0,
                avg_cache_read_cost_usd=0.0,
                total_cache_creation_cost_usd=0.0,
                avg_cache_creation_cost_usd=0.0,
                total_output_cost_usd=0.0,
                avg_output_cost_usd=0.0,
                total_cost_usd=0.0,
                avg_cost_usd=0.0,
                total_compression_input_cost_usd=0.0,
                avg_compression_input_cost_usd=0.0,
                total_compression_cache_read_cost_usd=0.0,
                avg_compression_cache_read_cost_usd=0.0,
                total_compression_cache_creation_cost_usd=0.0,
                avg_compression_cache_creation_cost_usd=0.0,
                total_compression_output_cost_usd=0.0,
                avg_compression_output_cost_usd=0.0,
                total_compression_cost_usd=0.0,
                avg_compression_cost_usd=0.0,
                trajectory_generation_time_seconds=trajectory_generation_time_seconds,
                timestamp=datetime.now().isoformat(),
            )

        total = len(metrics_list)
        aggregate_total = max(instance_count, total)
        successful = sum(1 for m in metrics_list if m.has_submission)

        # Token totals
        total_prompt = sum(m.total_prompt_tokens for m in metrics_list)
        total_completion = sum(m.total_completion_tokens for m in metrics_list)

        # API calls
        total_api_calls = sum(m.api_calls for m in metrics_list)

        # Compression time
        total_comp_time = sum(m.compression_time_ms for m in metrics_list)
        total_compressed_steps = sum(m.compressed_steps for m in metrics_list)
        total_compression_prompt_tokens = sum(m.compression_prompt_tokens for m in metrics_list)
        total_compression_completion_tokens = sum(
            m.compression_completion_tokens for m in metrics_list
        )
        total_compression_tokens = sum(m.compression_total_tokens for m in metrics_list)
        total_compression_cached_tokens = sum(m.compression_cached_tokens for m in metrics_list)

        # Cost per category
        total_input_cost = sum(m.input_cost_usd for m in metrics_list)
        total_cache_read_cost = sum(m.cache_read_cost_usd for m in metrics_list)
        total_cache_creation_cost = sum(m.cache_creation_cost_usd for m in metrics_list)
        total_output_cost = sum(m.output_cost_usd for m in metrics_list)
        total_cost = sum(m.total_cost_usd for m in metrics_list)

        total_compression_input_cost = sum(m.compression_input_cost_usd for m in metrics_list)
        total_compression_cache_read_cost = sum(
            m.compression_cache_read_cost_usd for m in metrics_list
        )
        total_compression_cache_creation_cost = sum(
            m.compression_cache_creation_cost_usd for m in metrics_list
        )
        total_compression_output_cost = sum(m.compression_output_cost_usd for m in metrics_list)
        total_compression_cost = sum(m.compression_total_cost_usd for m in metrics_list)

        return AggregatedMetrics(
            method=method,
            total_instances=aggregate_total,
            successful_submissions=successful,
            failed_submissions=aggregate_total - successful,
            avg_prompt_tokens=total_prompt / total,
            avg_completion_tokens=total_completion / total,
            avg_total_tokens=(total_prompt + total_completion) / total,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            avg_api_calls=total_api_calls / total,
            total_api_calls=total_api_calls,
            avg_compression_time_ms=total_comp_time / total,
            total_compression_time_ms=total_comp_time,
            avg_compressed_steps=total_compressed_steps / total,
            total_compressed_steps=total_compressed_steps,
            total_compression_prompt_tokens=total_compression_prompt_tokens,
            avg_compression_prompt_tokens=total_compression_prompt_tokens / total,
            total_compression_completion_tokens=total_compression_completion_tokens,
            avg_compression_completion_tokens=total_compression_completion_tokens / total,
            total_compression_tokens=total_compression_tokens,
            avg_compression_tokens=total_compression_tokens / total,
            total_compression_cached_tokens=total_compression_cached_tokens,
            avg_compression_cached_tokens=total_compression_cached_tokens / total,
            total_input_cost_usd=total_input_cost,
            avg_input_cost_usd=total_input_cost / total,
            total_cache_read_cost_usd=total_cache_read_cost,
            avg_cache_read_cost_usd=total_cache_read_cost / total,
            total_cache_creation_cost_usd=total_cache_creation_cost,
            avg_cache_creation_cost_usd=total_cache_creation_cost / total,
            total_output_cost_usd=total_output_cost,
            avg_output_cost_usd=total_output_cost / total,
            total_cost_usd=total_cost,
            avg_cost_usd=total_cost / total,
            total_compression_input_cost_usd=total_compression_input_cost,
            avg_compression_input_cost_usd=total_compression_input_cost / total,
            total_compression_cache_read_cost_usd=total_compression_cache_read_cost,
            avg_compression_cache_read_cost_usd=total_compression_cache_read_cost / total,
            total_compression_cache_creation_cost_usd=total_compression_cache_creation_cost,
            avg_compression_cache_creation_cost_usd=total_compression_cache_creation_cost / total,
            total_compression_output_cost_usd=total_compression_output_cost,
            avg_compression_output_cost_usd=total_compression_output_cost / total,
            total_compression_cost_usd=total_compression_cost,
            avg_compression_cost_usd=total_compression_cost / total,
            trajectory_generation_time_seconds=trajectory_generation_time_seconds,
            timestamp=datetime.now().isoformat(),
        )


def wilson_score_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    Thin wrapper over ``statsmodels.stats.proportion.proportion_confint`` with
    ``method='wilson'``. The Wilson interval is preferred over the normal
    approximation because it behaves well at the boundaries (0 and 1) and for
    moderate ``n``.

    Args:
        successes: Number of successes observed.
        n: Number of trials.
        confidence: Two-sided confidence level (default 0.95).

    Returns:
        ``(lower, upper)`` clamped to ``[0, 1]``.

    Raises:
        ValueError: If ``n <= 0`` or ``successes`` is outside ``[0, n]``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must satisfy 0 <= k <= n; got k={successes}, n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")

    lower, upper = proportion_confint(successes, n, alpha=1.0 - confidence, method="wilson")
    return float(lower), float(upper)


@dataclass
class PassAt1Stats:
    """PASS@1 point estimate with a Wilson confidence interval."""

    method: str
    n: int
    successes: int
    pass_at_1: float
    ci_low: float
    ci_high: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n": self.n,
            "successes": self.successes,
            "pass_at_1": self.pass_at_1,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
        }


@dataclass
class PairwiseSignificance:
    """McNemar exact-test result for a paired pass@1 comparison."""

    method_a: str
    method_b: str
    n: int  # Total paired instances compared.
    both_pass: int
    only_a: int
    only_b: int
    neither: int
    pass_a: float
    pass_b: float
    delta: float  # pass_a - pass_b
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "n": self.n,
            "both_pass": self.both_pass,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "neither": self.neither,
            "pass_a": self.pass_a,
            "pass_b": self.pass_b,
            "delta": self.delta,
            "p_value": self.p_value,
        }


def compute_pass_at_1_stats(
    resolved_ids: set[str],
    instance_ids: set[str],
    method: str,
    confidence: float = 0.95,
) -> PassAt1Stats:
    """Compute pass@1 and Wilson CI for a single method over a fixed universe.

    Args:
        resolved_ids: Instance IDs the method resolved (intersected with universe).
        instance_ids: Universe of instance IDs to evaluate against (denominator).
        method: Method label.
        confidence: Two-sided confidence level (default 0.95).

    Returns:
        ``PassAt1Stats`` with the proportion and CI bounds.
    """
    n = len(instance_ids)
    successes = len(resolved_ids & instance_ids)
    ci_low, ci_high = wilson_score_interval(successes, n, confidence)
    return PassAt1Stats(
        method=method,
        n=n,
        successes=successes,
        pass_at_1=successes / n,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
    )


def compute_pairwise_significance(
    method_a: str,
    resolved_a: set[str],
    method_b: str,
    resolved_b: set[str],
    instance_ids: set[str],
) -> PairwiseSignificance:
    """McNemar exact two-sided test for paired pass@1 between two methods.

    Args:
        method_a: Label for method A.
        resolved_a: IDs resolved by method A (intersected with universe).
        method_b: Label for method B.
        resolved_b: IDs resolved by method B (intersected with universe).
        instance_ids: Universe of instances used for the comparison.

    Returns:
        ``PairwiseSignificance`` with the contingency table and p-value.
    """
    a_set = resolved_a & instance_ids
    b_set = resolved_b & instance_ids
    both = len(a_set & b_set)
    only_a = len(a_set - b_set)
    only_b = len(b_set - a_set)
    n = len(instance_ids)
    neither = n - both - only_a - only_b
    # 2x2 table laid out as [[both_pass, only_a], [only_b, neither]];
    # McNemar's exact test only consumes the off-diagonal counts.
    table = np.array([[both, only_a], [only_b, neither]])
    p_value = float(mcnemar(table, exact=True).pvalue)
    return PairwiseSignificance(
        method_a=method_a,
        method_b=method_b,
        n=n,
        both_pass=both,
        only_a=only_a,
        only_b=only_b,
        neither=neither,
        pass_a=len(a_set) / n,
        pass_b=len(b_set) / n,
        delta=(len(a_set) - len(b_set)) / n,
        p_value=p_value,
    )


def _load_resolved_and_universe(report_path: Path) -> tuple[set[str], set[str]]:
    """Read a SWE-bench report.json and return (resolved_ids, submitted_ids).

    Args:
        report_path: Path to ``report.json`` produced by the SWE-bench harness.

    Returns:
        ``(resolved_ids, submitted_ids)`` as string sets.
    """
    payload = json.loads(report_path.read_text())
    return set(payload["resolved_ids"]), set(payload["submitted_ids"])


def compute_run_significance(
    run_dir: Path,
    target_method: str = "CoACT",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute pass@1 stats for every method and pairwise tests vs ``target_method``.

    For each method this reads ``<run_dir>/results/<method>/swebench/report.json``,
    builds a shared instance universe (intersection of ``submitted_ids`` across
    methods so the comparisons are paired on identical instances), and returns:

    * per-method pass@1 with Wilson CI computed over the shared universe;
    * pairwise McNemar exact-test results comparing ``target_method`` to every
      other method.

    Args:
        run_dir: Run directory containing ``results/<method>/swebench/report.json``.
        target_method: Method to compare against the rest (default ``"CoACT"``).
        confidence: Confidence level for the Wilson CI (default 0.95).

    Returns:
        Dict with keys ``run_id``, ``methods``, ``shared_instances``,
        ``per_method`` (list of ``PassAt1Stats.to_dict()``), and
        ``pairwise`` (list of ``PairwiseSignificance.to_dict()``).

    Raises:
        FileNotFoundError: If ``run_dir/results`` does not exist or a report is missing.
        ValueError: If ``target_method`` has no report in ``run_dir``.
    """
    results_dir = run_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {results_dir}")

    method_data: dict[str, tuple[set[str], set[str]]] = {}
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        report_path = entry / "swebench" / "report.json"
        if not report_path.exists():
            continue
        method_data[entry.name] = _load_resolved_and_universe(report_path)

    if target_method not in method_data:
        raise ValueError(
            f"target_method {target_method!r} has no report under {results_dir}; "
            f"available: {sorted(method_data)}"
        )

    shared_universe: set[str] = set.intersection(
        *(submitted for _resolved, submitted in method_data.values())
    )

    per_method = [
        compute_pass_at_1_stats(resolved, shared_universe, method, confidence).to_dict()
        for method, (resolved, _submitted) in method_data.items()
    ]

    target_resolved, _target_submitted = method_data[target_method]
    pairwise = [
        compute_pairwise_significance(
            target_method,
            target_resolved,
            other,
            resolved,
            shared_universe,
        ).to_dict()
        for other, (resolved, _submitted) in method_data.items()
        if other != target_method
    ]

    return {
        "run_id": run_dir.name,
        "target_method": target_method,
        "confidence": confidence,
        "methods": sorted(method_data),
        "shared_instances": len(shared_universe),
        "per_method": per_method,
        "pairwise": pairwise,
    }


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    rng_seed: int = 0,
    method: str = "BCa",
) -> tuple[float, float]:
    """Bootstrap confidence interval for the sample mean.

    Args:
        values: One-dimensional numeric sample.
        confidence: Two-sided confidence level (default 0.95).
        n_resamples: Number of bootstrap resamples.
        rng_seed: Seed for the bootstrap RNG (deterministic output).
        method: Bootstrap CI method passed to ``scipy.stats.bootstrap``
            (``"BCa"``, ``"percentile"``, or ``"basic"``).

    Returns:
        ``(lower, upper)`` confidence bounds.

    Raises:
        ValueError: If ``values`` is empty or ``confidence`` is invalid.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if arr.size == 1 or np.allclose(arr, arr[0]):
        mean = float(arr.mean())
        return mean, mean
    rng = np.random.default_rng(rng_seed)
    result = _scipy_stats.bootstrap(
        (arr,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method=method,
        random_state=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


@dataclass
class CostStats:
    """Per-method cost summary with a bootstrap CI for the mean."""

    method: str
    n: int
    mean: float
    median: float
    std: float
    total: float
    ci_low: float
    ci_high: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "total": self.total,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
        }


@dataclass
class PairwiseCostSignificance:
    """Paired comparison of per-instance costs (method_a − method_b)."""

    method_a: str
    method_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float  # mean(cost_a - cost_b) over paired instances
    median_diff: float
    diff_ci_low: float
    diff_ci_high: float
    wilcoxon_pvalue: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_diff": self.mean_diff,
            "median_diff": self.median_diff,
            "diff_ci_low": self.diff_ci_low,
            "diff_ci_high": self.diff_ci_high,
            "wilcoxon_pvalue": self.wilcoxon_pvalue,
        }


def compute_cost_stats(
    method: str,
    costs: dict[str, float],
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> CostStats:
    """Summarize a per-instance cost mapping with a bootstrap mean CI.

    Args:
        method: Method label.
        costs: Mapping ``instance_id -> cost`` (use whichever cost is desired).
        confidence: Two-sided confidence level for the bootstrap CI.
        rng_seed: Deterministic seed for the bootstrap RNG.

    Returns:
        ``CostStats`` with mean / median / std / total and the CI.
    """
    arr = np.asarray(list(costs.values()), dtype=float)
    if arr.size == 0:
        raise ValueError(f"method {method!r}: costs is empty")
    ci_low, ci_high = bootstrap_mean_ci(arr, confidence=confidence, rng_seed=rng_seed)
    return CostStats(
        method=method,
        n=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        total=float(arr.sum()),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
    )


def compute_pairwise_cost_significance(
    method_a: str,
    costs_a: dict[str, float],
    method_b: str,
    costs_b: dict[str, float],
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> PairwiseCostSignificance:
    """Wilcoxon signed-rank + bootstrap CI for paired per-instance cost diffs.

    Pairs are formed by ``set(costs_a) & set(costs_b)``. Instances missing from
    either side are dropped (fail-fast: at least one shared instance required).

    Args:
        method_a: Label for method A.
        costs_a: Per-instance costs for method A.
        method_b: Label for method B.
        costs_b: Per-instance costs for method B.
        confidence: Two-sided confidence level for the bootstrap Δ-mean CI.
        rng_seed: Deterministic seed for the bootstrap RNG.

    Returns:
        ``PairwiseCostSignificance`` with paired summary and Wilcoxon p-value.
        Wilcoxon p-value is ``1.0`` if all pairwise diffs are zero (degenerate).
    """
    shared = sorted(set(costs_a) & set(costs_b))
    if not shared:
        raise ValueError(f"No shared instances between {method_a!r} and {method_b!r}")
    a = np.array([costs_a[i] for i in shared], dtype=float)
    b = np.array([costs_b[i] for i in shared], dtype=float)
    diffs = a - b

    diff_ci_low, diff_ci_high = bootstrap_mean_ci(
        diffs, confidence=confidence, rng_seed=rng_seed
    )
    if np.allclose(diffs, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(_scipy_stats.wilcoxon(diffs, zero_method="wilcox").pvalue)

    return PairwiseCostSignificance(
        method_a=method_a,
        method_b=method_b,
        n=len(shared),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(diffs.mean()),
        median_diff=float(np.median(diffs)),
        diff_ci_low=diff_ci_low,
        diff_ci_high=diff_ci_high,
        wilcoxon_pvalue=wilcoxon_p,
    )


def _load_per_instance_costs(
    trajectory_dir: Path,
    model: str,
    cost_field: str,
) -> dict[str, float]:
    """Walk a method's trajectory directory and extract per-instance costs.

    Args:
        trajectory_dir: Directory containing ``<instance_id>.json`` trajectories.
        model: LiteLLM/pricing model identifier for ``MetricsExtractor.extract``.
        cost_field: Attribute on ``TrajectoryMetrics`` to read (e.g.
            ``"total_cost_usd"``, ``"input_cost_usd"``, ``"output_cost_usd"``).

    Returns:
        Mapping ``instance_id -> cost``.

    Raises:
        AttributeError: If ``cost_field`` is not a TrajectoryMetrics attribute.
    """
    out: dict[str, float] = {}
    for path in sorted(trajectory_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        metrics = MetricsExtractor.extract(payload, model=model)
        out[metrics.instance_id] = float(getattr(metrics, cost_field))
    return out


def compute_run_cost_significance(
    run_dir: Path,
    model: str,
    target_method: str = "CoACT",
    cost_field: str = "total_cost_usd",
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Compute per-method cost stats and paired tests vs ``target_method``.

    For each method the per-instance cost is extracted from its trajectory
    directory via ``MetricsExtractor.extract``. Per-method summaries use all
    instances available for that method; pairwise tests pair on the
    intersection of instance IDs.

    Args:
        run_dir: Run directory containing ``trajectories/<method>/<id>.json``.
        model: Model identifier used for cost lookup (must match the active
            ``PricingConfig`` installed via ``set_active_pricing``).
        target_method: Method to compare against the rest (default ``"CoACT"``).
        cost_field: Which cost attribute to use (default ``"total_cost_usd"``;
            also valid: ``"input_cost_usd"``, ``"output_cost_usd"``,
            ``"compression_total_cost_usd"``).
        confidence: Two-sided confidence level for the bootstrap CIs.
        rng_seed: Deterministic seed for the bootstrap RNG.

    Returns:
        Dict with ``run_id``, ``cost_field``, ``methods``, ``per_method``
        (list of ``CostStats.to_dict()``), and ``pairwise`` (list of
        ``PairwiseCostSignificance.to_dict()``).

    Raises:
        FileNotFoundError: If ``trajectories/`` or ``target_method`` is missing.
    """
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        raise FileNotFoundError(f"trajectories directory not found: {traj_root}")

    method_costs: dict[str, dict[str, float]] = {}
    for entry in sorted(traj_root.iterdir()):
        if not entry.is_dir():
            continue
        method_costs[entry.name] = _load_per_instance_costs(entry, model, cost_field)

    if target_method not in method_costs:
        raise FileNotFoundError(
            f"target_method {target_method!r} not found under {traj_root}; "
            f"available: {sorted(method_costs)}"
        )

    per_method = [
        compute_cost_stats(method, costs, confidence=confidence, rng_seed=rng_seed).to_dict()
        for method, costs in method_costs.items()
    ]

    target_costs = method_costs[target_method]
    pairwise = [
        compute_pairwise_cost_significance(
            target_method,
            target_costs,
            other,
            costs,
            confidence=confidence,
            rng_seed=rng_seed,
        ).to_dict()
        for other, costs in method_costs.items()
        if other != target_method
    ]

    return {
        "run_id": run_dir.name,
        "target_method": target_method,
        "cost_field": cost_field,
        "confidence": confidence,
        "methods": sorted(method_costs),
        "per_method": per_method,
        "pairwise": pairwise,
    }
