"""Selection helpers for raw rollout samples used in SFT training."""

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

RawRolloutRecord = dict[str, Any]
TrainingExample = dict[str, str]
SelectionMode = Literal["reward", "length_only", "action_only"]

# Canonical "unchanged" completion produced by the compressor output parser.
UNCHANGED_COMPLETION = '{"type":"unchanged","content":null}'


def _build_unchanged_fallback_record(
    group_records: Sequence[RawRolloutRecord],
) -> RawRolloutRecord:
    """Create an ``unchanged`` fallback record for a group whose rollouts all failed.

    The fallback teaches the compressor to leave the output alone when the
    signal (CFQ, tool output) is too weak for any compression to preserve agent
    behavior. Picks the highest-action_reward member as the carrier so group
    metadata stays consistent.

    Args:
        group_records: Rollout records sharing the same (trajectory_id, step_id).

    Returns:
        New rollout record with completion replaced by ``UNCHANGED_COMPLETION``.

    Raises:
        ValueError: If ``group_records`` is empty.
    """
    if not group_records:
        raise ValueError("Cannot build fallback record from empty group")

    carrier = max(group_records, key=lambda item: float(item["action_reward"]))
    return {
        **carrier,
        "completion": UNCHANGED_COMPLETION,
        "action_reward": 1.0,
        "length_reward": 0.0,
        "inferred_action": carrier["inferred_action"],
        "is_unchanged_fallback": True,
    }


def select_rollout_samples(
    records: Sequence[RawRolloutRecord],
    *,
    top_k: int,
    min_similarity: float,
    inject_unchanged_fallback: bool = False,
    selection_mode: SelectionMode = "reward",
) -> list[RawRolloutRecord]:
    """Select training candidates from raw rollout records.

    In ``reward`` mode, candidates below ``min_similarity`` are first dropped
    (the behavior-preservation quality gate on ``action_reward``). Among the
    surviving candidates the ``top_k`` with the highest ``length_reward`` are
    kept, with ``action_reward`` breaking ties.

    In ``length_only`` mode, the action gate is disabled and candidates are
    ranked only by ``length_reward``. In ``action_only`` mode, the action gate
    is kept but candidates are ranked only by ``action_reward``.

    When ``inject_unchanged_fallback`` is set and a group has no candidate
    meeting the threshold, one synthetic ``unchanged`` completion is added
    instead so the model still sees that step during training.

    Args:
        records: Raw rollout records loaded from JSONL.
        top_k: Number of shortest passing samples to keep per trajectory step.
        min_similarity: Minimum action_reward threshold (quality gate).
        inject_unchanged_fallback: When ``True``, replace empty groups with a
            single ``{"type":"unchanged","content":null}`` completion.
        selection_mode: Reward-ablation selection mode.

    Returns:
        Selected raw rollout records.

    Raises:
        ValueError: If ``selection_mode`` is unknown.
    """
    if selection_mode not in {"reward", "length_only", "action_only"}:
        raise ValueError(f"Unknown rollout selection mode: {selection_mode}")

    grouped_records: dict[tuple[str, int], list[RawRolloutRecord]] = defaultdict(list)
    for record in records:
        group_key = (record["trajectory_id"], record["step_id"])
        grouped_records[group_key].append(record)

    selected_records: list[RawRolloutRecord] = []
    for _, group in grouped_records.items():
        if selection_mode == "length_only":
            passing_records = list(group)
        else:
            passing_records = [
                item for item in group if float(item["action_reward"]) >= min_similarity
            ]

        if passing_records:
            if selection_mode == "action_only":
                ranked_records = sorted(
                    passing_records,
                    key=lambda item: float(item["action_reward"]),
                    reverse=True,
                )
            else:
                ranked_records = sorted(
                    passing_records,
                    key=lambda item: (
                        float(item["length_reward"]),
                        float(item["action_reward"]),
                    ),
                    reverse=True,
                )
            selected_records.extend(ranked_records[:top_k])
        elif inject_unchanged_fallback:
            selected_records.append(_build_unchanged_fallback_record(group))

    return selected_records


def split_rollout_records_by_group(
    records: Sequence[RawRolloutRecord],
    *,
    eval_split_ratio: float,
    seed: int,
) -> tuple[list[RawRolloutRecord], list[RawRolloutRecord]]:
    """Split raw rollout records by trajectory-step group.

    Args:
        records: Raw rollout records loaded from JSONL.
        eval_split_ratio: Fraction of groups allocated to evaluation.
        seed: Random seed for reproducible group shuffling.

    Returns:
        Tuple of ``(train_records, eval_records)``.
    """
    if not records:
        return [], []

    if eval_split_ratio == 0:
        return list(records), []

    grouped_records: dict[tuple[str, int], list[RawRolloutRecord]] = defaultdict(list)
    for record in records:
        group_key = (record["trajectory_id"], record["step_id"])
        grouped_records[group_key].append(record)

    group_keys = list(grouped_records.keys())
    random.Random(seed).shuffle(group_keys)

    eval_group_count = int(round(len(group_keys) * eval_split_ratio))
    eval_group_count = min(max(eval_group_count, 1), len(group_keys) - 1)
    eval_group_keys = set(group_keys[:eval_group_count])

    train_records: list[RawRolloutRecord] = []
    eval_records: list[RawRolloutRecord] = []
    for group_key, group_records in grouped_records.items():
        if group_key in eval_group_keys:
            eval_records.extend(group_records)
        else:
            train_records.extend(group_records)

    return train_records, eval_records


def create_prompt_completion_examples(
    records: Sequence[RawRolloutRecord],
) -> list[TrainingExample]:
    """Convert selected rollout records into prompt/completion training pairs.

    Args:
        records: Selected rollout records.

    Returns:
        List of training examples containing only prompt and completion fields.
    """
    examples: list[TrainingExample] = []
    for record in records:
        examples.append(
            {
                "prompt": record["prompt"],
                "completion": record["completion"],
            }
        )
    return examples
