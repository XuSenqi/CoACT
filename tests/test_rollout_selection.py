"""Tests for raw rollout selection helpers."""

from src.compression.rollout_selection import (
    UNCHANGED_COMPLETION,
    create_prompt_completion_examples,
    select_rollout_samples,
    split_rollout_records_by_group,
)


def _make_record(
    *,
    prompt: str,
    completion: str,
    total_reward: float,
    trajectory_id: str,
    step_id: int,
) -> dict:
    """Create a raw rollout record for selection tests."""
    return {
        "prompt": prompt,
        "completion": completion,
        "action_reward": total_reward,
        "length_reward": 0.0,
        "total_reward": total_reward,
        "inferred_action": "grep TODO app.py",
        "ground_truth": ["grep TODO app.py"],
        "trajectory_id": trajectory_id,
        "step_id": step_id,
    }


def test_select_rollout_samples_applies_top_k_per_step() -> None:
    """Top-k selection should be applied within each trajectory step group."""
    records = [
        _make_record(
            prompt="Prompt 1",
            completion="keep-high",
            total_reward=0.95,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt 1",
            completion="keep-mid",
            total_reward=0.90,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt 1",
            completion="drop-low-rank",
            total_reward=0.85,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt 2",
            completion="keep-step-2",
            total_reward=0.93,
            trajectory_id="traj-1",
            step_id=2,
        ),
    ]

    selected = select_rollout_samples(
        records,
        top_k=2,
        min_similarity=0.8,
    )

    selected_completions = {record["completion"] for record in selected}
    assert selected_completions == {"keep-high", "keep-mid", "keep-step-2"}


def test_select_rollout_samples_uses_action_reward() -> None:
    """Selection ranks and thresholds by action_reward (legacy total_reward unused)."""
    records = [
        {
            **_make_record(
                prompt="Prompt 1",
                completion="keep-by-action",
                total_reward=0.75,
                trajectory_id="traj-1",
                step_id=1,
            ),
            "action_reward": 0.99,
        },
        {
            **_make_record(
                prompt="Prompt 1",
                completion="drop-by-action",
                total_reward=0.90,
                trajectory_id="traj-1",
                step_id=1,
            ),
            "action_reward": 0.64,
        },
    ]

    selected = select_rollout_samples(
        records,
        top_k=1,
        min_similarity=0.8,
    )

    assert [record["completion"] for record in selected] == ["keep-by-action"]


def test_select_prefers_higher_length_reward_among_passing() -> None:
    """Among action-passing candidates, top-k keeps the highest length_reward."""
    records = [
        {
            **_make_record(
                prompt="P",
                completion="long-high-action",
                total_reward=0.99,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.99,
            "length_reward": -0.5,
        },
        {
            **_make_record(
                prompt="P",
                completion="short",
                total_reward=0.85,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.85,
            "length_reward": 0.5,
        },
        {
            **_make_record(
                prompt="P",
                completion="below-gate",
                total_reward=0.50,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.50,
            "length_reward": 0.5,
        },
    ]

    selected = select_rollout_samples(records, top_k=1, min_similarity=0.8)

    # below-gate dropped (action 0.5 < 0.8); among the passing pair, "short"
    # wins on length_reward despite a lower action_reward than the long one.
    assert [record["completion"] for record in selected] == ["short"]


def test_select_breaks_length_ties_by_action_reward() -> None:
    """Equal length_reward is broken deterministically by higher action_reward."""
    records = [
        {
            **_make_record(
                prompt="P", completion="a", total_reward=0.7, trajectory_id="t", step_id=1
            ),
            "action_reward": 0.7,
            "length_reward": 0.0,
        },
        {
            **_make_record(
                prompt="P", completion="b", total_reward=0.9, trajectory_id="t", step_id=1
            ),
            "action_reward": 0.9,
            "length_reward": 0.0,
        },
    ]

    selected = select_rollout_samples(records, top_k=1, min_similarity=0.6)

    assert [record["completion"] for record in selected] == ["b"]


def test_length_only_selection_ignores_action_gate() -> None:
    """RQ3 w/o AP mode should select by length_reward without action filtering."""
    records = [
        {
            **_make_record(
                prompt="P",
                completion="long-action-preserving",
                total_reward=0.99,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.99,
            "length_reward": -0.5,
        },
        {
            **_make_record(
                prompt="P",
                completion="short-low-action",
                total_reward=0.10,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.10,
            "length_reward": 0.5,
        },
    ]

    selected = select_rollout_samples(
        records,
        top_k=1,
        min_similarity=0.8,
        selection_mode="length_only",
    )

    assert [record["completion"] for record in selected] == ["short-low-action"]


def test_action_only_selection_ignores_length_reward() -> None:
    """RQ3 w/o LR mode should filter by action and rank by action_reward."""
    records = [
        {
            **_make_record(
                prompt="P",
                completion="short-lower-action",
                total_reward=0.85,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.85,
            "length_reward": 0.5,
        },
        {
            **_make_record(
                prompt="P",
                completion="long-higher-action",
                total_reward=0.99,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.99,
            "length_reward": -0.5,
        },
        {
            **_make_record(
                prompt="P",
                completion="short-below-gate",
                total_reward=0.20,
                trajectory_id="t",
                step_id=1,
            ),
            "action_reward": 0.20,
            "length_reward": 0.5,
        },
    ]

    selected = select_rollout_samples(
        records,
        top_k=1,
        min_similarity=0.8,
        selection_mode="action_only",
    )

    assert [record["completion"] for record in selected] == ["long-higher-action"]


def test_unknown_selection_mode_fails_fast() -> None:
    """Unknown selection modes should fail at the point of use."""
    records = [
        _make_record(
            prompt="P",
            completion="x",
            total_reward=0.9,
            trajectory_id="t",
            step_id=1,
        )
    ]

    try:
        select_rollout_samples(
            records,
            top_k=1,
            min_similarity=0.8,
            selection_mode="unknown",
        )
    except ValueError as exc:
        assert "Unknown rollout selection mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown selection mode")


def test_select_rollout_samples_filters_threshold_without_deduplicating() -> None:
    """Selection should drop low-reward records while preserving duplicates."""
    records = [
        _make_record(
            prompt="Prompt",
            completion="duplicate completion",
            total_reward=0.95,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt",
            completion="duplicate completion",
            total_reward=0.94,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt",
            completion="below-threshold",
            total_reward=0.30,
            trajectory_id="traj-1",
            step_id=1,
        ),
    ]

    selected = select_rollout_samples(
        records,
        top_k=3,
        min_similarity=0.8,
    )

    assert len(selected) == 2
    assert [record["completion"] for record in selected] == [
        "duplicate completion",
        "duplicate completion",
    ]


def test_select_rollout_samples_injects_unchanged_when_group_fails_threshold() -> None:
    """All-bad groups contribute one synthetic ``unchanged`` completion."""
    records = [
        _make_record(
            prompt="Prompt failed",
            completion='{"type":"code","content":["1-5:irrelevant"]}',
            total_reward=0.20,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt failed",
            completion='{"type":"plain","content":"..."}',
            total_reward=0.05,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt good",
            completion='{"type":"code","content":["1-5:ok"]}',
            total_reward=0.95,
            trajectory_id="traj-1",
            step_id=2,
        ),
    ]

    selected = select_rollout_samples(
        records,
        top_k=4,
        min_similarity=0.3,
        inject_unchanged_fallback=True,
    )

    completions = [record["completion"] for record in selected]
    assert UNCHANGED_COMPLETION in completions
    assert '{"type":"code","content":["1-5:ok"]}' in completions

    fallback = next(record for record in selected if record["completion"] == UNCHANGED_COMPLETION)
    assert fallback["trajectory_id"] == "traj-1"
    assert fallback["step_id"] == 1
    assert fallback["prompt"] == "Prompt failed"
    assert fallback["action_reward"] == 1.0
    assert fallback["is_unchanged_fallback"] is True


def test_select_rollout_samples_fallback_disabled_drops_all_bad_group() -> None:
    """With fallback off, all-bad groups are dropped entirely."""
    records = [
        _make_record(
            prompt="Prompt",
            completion="low-1",
            total_reward=0.10,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt",
            completion="low-2",
            total_reward=0.05,
            trajectory_id="traj-1",
            step_id=1,
        ),
    ]

    selected = select_rollout_samples(
        records,
        top_k=4,
        min_similarity=0.3,
        inject_unchanged_fallback=False,
    )

    assert selected == []


def test_select_rollout_samples_fallback_skipped_when_group_has_passing_record() -> None:
    """Groups that have any passing record never get a synthetic fallback."""
    records = [
        _make_record(
            prompt="Prompt",
            completion="passes",
            total_reward=0.80,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt",
            completion="fails",
            total_reward=0.10,
            trajectory_id="traj-1",
            step_id=1,
        ),
    ]

    selected = select_rollout_samples(
        records,
        top_k=4,
        min_similarity=0.3,
        inject_unchanged_fallback=True,
    )

    completions = [record["completion"] for record in selected]
    assert completions == ["passes"]
    assert UNCHANGED_COMPLETION not in completions


def test_split_rollout_records_by_group_keeps_group_boundaries() -> None:
    """Group-level split should never divide one step's candidates."""
    records = [
        _make_record(
            prompt="Prompt 1",
            completion="step1-a",
            total_reward=0.95,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt 1",
            completion="step1-b",
            total_reward=0.90,
            trajectory_id="traj-1",
            step_id=1,
        ),
        _make_record(
            prompt="Prompt 2",
            completion="step2-a",
            total_reward=0.88,
            trajectory_id="traj-1",
            step_id=2,
        ),
        _make_record(
            prompt="Prompt 2",
            completion="step2-b",
            total_reward=0.84,
            trajectory_id="traj-1",
            step_id=2,
        ),
    ]

    train_records, eval_records = split_rollout_records_by_group(
        records,
        eval_split_ratio=0.5,
        seed=42,
    )

    train_groups = {(record["trajectory_id"], record["step_id"]) for record in train_records}
    eval_groups = {(record["trajectory_id"], record["step_id"]) for record in eval_records}
    assert train_groups.isdisjoint(eval_groups)
    assert train_groups | eval_groups == {("traj-1", 1), ("traj-1", 2)}


def test_build_training_examples_keeps_prompt_and_completion() -> None:
    """Training example conversion should keep only prompt and completion."""
    examples = create_prompt_completion_examples(
        [
            _make_record(
                prompt="Prompt",
                completion="Output",
                total_reward=0.9,
                trajectory_id="traj-1",
                step_id=1,
            )
        ]
    )

    assert examples == [{"prompt": "Prompt", "completion": "Output"}]


def test_namespaced_ids_keep_sources_in_separate_groups() -> None:
    """offpolicy:/dagger: prefixed ids with the same step never share a group."""
    records = [
        _make_record(
            prompt="Off-policy step",
            completion="offpolicy-best",
            total_reward=0.95,
            trajectory_id="offpolicy:inst-1",
            step_id=7,
        ),
        _make_record(
            prompt="Off-policy step",
            completion="offpolicy-worst",
            total_reward=0.10,
            trajectory_id="offpolicy:inst-1",
            step_id=7,
        ),
        _make_record(
            prompt="Dagger step",
            completion="dagger-best",
            total_reward=0.92,
            trajectory_id="dagger:inst-1",
            step_id=7,
        ),
        _make_record(
            prompt="Dagger step",
            completion="dagger-worst",
            total_reward=0.05,
            trajectory_id="dagger:inst-1",
            step_id=7,
        ),
    ]

    # top_k=1 per group: each source contributes its own best. If the two
    # sources shared a (trajectory_id, step_id) group, only one overall best
    # would survive — this asserts the namespacing keeps them apart.
    selected = select_rollout_samples(records, top_k=1, min_similarity=0.0)
    assert {r["completion"] for r in selected} == {"offpolicy-best", "dagger-best"}

    # Group split also keys on the namespaced id: the two sources are two
    # distinct groups, so a train/eval split never mixes a source across sides.
    train, eval_ = split_rollout_records_by_group(records, eval_split_ratio=0.5, seed=0)
    train_ids = {r["trajectory_id"] for r in train}
    eval_ids = {r["trajectory_id"] for r in eval_}
    assert train_ids and eval_ids
    assert train_ids.isdisjoint(eval_ids)
