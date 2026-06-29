"""Tests for DAGGER SFT data preparation helpers."""

from __future__ import annotations

import asyncio

import pytest

from scripts import prepare_dagger_data
from src.agent.trajectory_parser import ToolOutput, TrajectoryStep, parse_trajectory


def _make_step(message_index: int) -> TrajectoryStep:
    """Create a minimal assistant step for DAGGER helper tests."""
    return TrajectoryStep(
        goal="Fix a bug",
        message_index=message_index,
        tool_output=(ToolOutput(returncode=0, output="compressed output"),),
        reasoning="Inspect output",
        tool_call=(f"grep step{message_index} app.py",),
        context_focus_question=("Inspect relevant output.",),
    )


def test_find_compressed_steps_skips_multi_tool_call_steps() -> None:
    """DAGGER data prep should match off-policy single-tool-step support."""
    messages = [
        {"role": "user", "content": "Fix bug"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "git status"}'},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "compressed first",
            "extra": {"original_tool_output": "original first"},
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": "compressed second",
            "extra": {"original_tool_output": "original second"},
        },
    ]
    trajectory = parse_trajectory({"info": {"instance_id": "traj-1"}, "messages": messages})

    compressed_steps = prepare_dagger_data.find_compressed_steps(trajectory, messages)

    assert compressed_steps == []


@pytest.mark.asyncio
async def test_process_pending_steps_runs_concurrently_and_isolates_failures() -> None:
    """Pending DAGGER steps should run concurrently without failing the whole batch."""
    active_workers = 0
    max_active_workers = 0
    lock = asyncio.Lock()

    class FakeProcessor:
        async def process_step(
            self,
            *,
            trajectory_id: str,
            deploy_messages: list[dict],
            asst_step: TrajectoryStep,
            first_tool_idx: int,
            original_tool_output: str,
        ) -> list[dict]:
            del trajectory_id, deploy_messages, first_tool_idx, original_tool_output
            nonlocal active_workers, max_active_workers
            async with lock:
                active_workers += 1
                max_active_workers = max(max_active_workers, active_workers)
            await asyncio.sleep(0.01)
            async with lock:
                active_workers -= 1
            if asst_step.message_index == 2:
                raise RuntimeError("boom")
            return [{"step_id": asst_step.message_index}]

    pending = [
        (_make_step(1), 10, ["original one"]),
        (_make_step(2), 20, ["original two"]),
        (_make_step(3), 30, ["original three"]),
    ]

    results = await prepare_dagger_data.process_pending_steps(
        processor=FakeProcessor(),
        trajectory_id="dagger:traj-parallel",
        deploy_messages=[{"role": "user", "content": "x"}],
        pending=pending,
        step_max_workers=2,
    )

    assert max_active_workers == 2
    assert [result.step.message_index for result in results] == [1, 2, 3]
    assert [result.failed for result in results] == [False, True, False]
    assert [result.records for result in results] == [
        [{"step_id": 1}],
        [],
        [{"step_id": 3}],
    ]
