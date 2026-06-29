"""Shared agent-natural anchor reward scoring.

Both the off-policy SFT data prep (:mod:`src.compression.data_preparation`) and
the on-policy DAGGER prep (``scripts/prepare_dagger_data.py``) score a
compression candidate by how well the agent's *inferred* next action matches a
set of *anchor* actions sampled from the agent on the uncompressed context.

This module owns that computation. It is agnostic to how the message prefix was
built: each caller passes a ready ``(prefix_messages, step_index, anchor_runner)``
triple, so the off-policy path can hand its raw trajectory while the DAGGER path
hands its original-output-swapped deploy prefix.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agent.miniswe_runner import MiniSWERunner
from src.reward.bash_similarity import compute_command_similarity

if TYPE_CHECKING:
    from src.config.config import DataPreparationConfig

VALID_AGGREGATORS = frozenset({"mean", "max", "top3", "count"})


@dataclass(frozen=True)
class AnchorAggregator:
    """How to collapse per-anchor similarities into a single reward.

    Attributes:
        kind: One of ``mean``, ``max``, ``top3``, ``count``.
        count_threshold: Similarity threshold used by the ``count`` aggregator.
    """

    kind: str = "top3"
    count_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in VALID_AGGREGATORS:
            raise ValueError(
                f"anchor aggregator must be one of {sorted(VALID_AGGREGATORS)}, got {self.kind!r}"
            )
        if not isinstance(self.count_threshold, (int, float)) or not (
            0.0 <= self.count_threshold <= 1.0
        ):
            raise ValueError(
                f"anchor count_threshold must be between 0 and 1, got {self.count_threshold}"
            )

    @classmethod
    def from_config(cls, data_prep_cfg: DataPreparationConfig) -> AnchorAggregator:
        """Build from a config exposing ``anchor_aggregator``/``anchor_count_threshold``."""
        return cls(
            kind=data_prep_cfg.anchor_aggregator,
            count_threshold=data_prep_cfg.anchor_count_threshold,
        )


def aggregate_similarities(sims: list[float], aggregator: AnchorAggregator) -> float:
    """Collapse per-anchor similarities into one reward in ``[0, 1]``.

    Args:
        sims: Per-anchor similarity scores.
        aggregator: The aggregation strategy.

    Returns:
        The aggregated reward, or ``0.0`` when ``sims`` is empty.
    """
    if not sims:
        return 0.0
    if aggregator.kind == "mean":
        return statistics.mean(sims)
    if aggregator.kind == "max":
        return max(sims)
    if aggregator.kind == "top3":
        return statistics.mean(sorted(sims, reverse=True)[: min(3, len(sims))])
    if aggregator.kind == "count":
        return sum(1 for s in sims if s >= aggregator.count_threshold) / len(sims)
    raise ValueError(f"Unknown aggregator: {aggregator.kind}")  # pragma: no cover


def score_candidate_action(
    inferred_action: str,
    anchor_set: list[str],
    aggregator: AnchorAggregator,
) -> float:
    """Score one inferred action against the anchor set.

    Args:
        inferred_action: The agent's next action under the compressed candidate.
        anchor_set: Agent-natural next actions on the uncompressed context.
        aggregator: How to collapse the per-anchor similarities.

    Returns:
        The aggregated similarity reward in ``[0, 1]``.
    """
    sims = [compute_command_similarity(inferred_action, anchor) for anchor in anchor_set]
    return aggregate_similarities(sims, aggregator)


async def sample_anchor_set(
    anchor_runner: MiniSWERunner,
    prefix_messages: list[dict[str, Any]],
    step_index: int,
    num_samples: int,
    max_workers: int,
) -> list[str]:
    """Sample the agent's natural next action ``num_samples`` times.

    Each sample continues from ``step_index`` on ``prefix_messages`` using the
    untouched (original) tool output, at the anchor runner's temperature. Blank
    and ``None`` completions are dropped.

    Args:
        anchor_runner: A runner configured at the anchor sampling temperature.
        prefix_messages: The message prefix to continue from (caller-built).
        step_index: Index of the assistant step to continue from.
        num_samples: Number of anchor samples to draw.
        max_workers: Maximum concurrent runner calls.

    Returns:
        The non-empty sampled commands (length ``<= num_samples``).
    """
    semaphore = asyncio.Semaphore(max_workers)

    async def one() -> str | None:
        async with semaphore:
            return await asyncio.to_thread(
                anchor_runner.continue_from_step,
                trajectory_messages=prefix_messages,
                step_index=step_index,
                compressed_output="",
                returncode=0,
                exception_info=None,
                use_original_output=True,
            )

    results = await asyncio.gather(*(one() for _ in range(num_samples)))
    return [r for r in results if isinstance(r, str) and r.strip()]
