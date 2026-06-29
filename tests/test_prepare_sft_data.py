"""Tests for raw rollout generation in `src.compression.data_preparation`."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.compression.data_preparation as data_preparation
from src.agent.trajectory_parser import ToolOutput, TrajectoryStep
from src.compression.compression_filter import CompressionFilter, CompressionFilterReason
from src.config.config import Config
from src.reward.bash_similarity import BashParseError


class _FakeTokenizer:
    """Fake tokenizer for testing token counting."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Return a token count based on whitespace splitting."""
        del add_special_tokens
        return list(range(len(text.split())))


@pytest.fixture
def config() -> Config:
    """Load the unified configuration."""
    return Config.load("config/config.yaml")


@pytest.fixture
def pipeline(config: Config):
    """Create a DataPreparationPipeline with mocked dependencies."""
    compression_filter = CompressionFilter(
        config.sft.data_preparation.skip_compression_max_tokens, _FakeTokenizer()
    )
    with patch.object(data_preparation, "acompletion", return_value=AsyncMock()):
        with patch.object(data_preparation, "MiniSWERunner", return_value=MagicMock()):
            with patch.object(
                data_preparation.CompressionFilter,
                "build",
                return_value=compression_filter,
            ):
                yield data_preparation.DataPreparationPipeline(config)


def _make_step(command: str, output: str) -> TrajectoryStep:
    """Create a mock TrajectoryStep for testing."""
    return TrajectoryStep(
        goal="Fix a bug",
        message_index=1,
        tool_output=(ToolOutput(returncode=0, output=output),),
        reasoning="Inspect the repository state",
        tool_call=(command,),
        context_focus_question=("Find the exact information needed from this output.",),
    )


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records from disk."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _make_rollout_sample(
    compressed_text: str,
    keep_original: bool,
    inferred_action: str,
    trajectory_id: str,
    normalized_completion: str | None = None,
    valid_response: bool = True,
) -> data_preparation.RolloutSample:
    """Create a rollout sample returned by `_infer_sample_action`."""
    if normalized_completion is None:
        if keep_original:
            normalized_completion = json.dumps(
                {
                    "type": "unchanged",
                    "content": None,
                },
                separators=(",", ":"),
            )
        else:
            normalized_completion = compressed_text
    return data_preparation.RolloutSample(
        compressed_text=compressed_text,
        effective_text=compressed_text,
        normalized_completion=normalized_completion,
        keep_original=keep_original,
        valid_response=valid_response,
        inferred_action=inferred_action,
        step_id=1,
        trajectory_id=trajectory_id,
    )


class TestDataPreparationFiltering:
    """Tests for DataPreparationPipeline.process_step filtering logic."""

    @pytest.mark.asyncio
    async def test_process_trajectory_runs_steps_concurrently_with_limit(
        self,
        pipeline,
        tmp_path: Path,
    ) -> None:
        """Trajectory processing should run step workers concurrently with a limit."""
        steps = [
            replace(_make_step("grep one app.py", "long output " * 200), message_index=1),
            replace(_make_step("grep two app.py", "long output " * 200), message_index=2),
            replace(_make_step("grep three app.py", "long output " * 200), message_index=3),
        ]
        trajectory = SimpleNamespace(instance_id="traj-parallel", steps=steps)
        trajectory_path = tmp_path / "traj-parallel.json"
        trajectory_path.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}))
        pipeline.step_max_workers = 2

        active_workers = 0
        max_active_workers = 0
        lock = asyncio.Lock()

        async def fake_process_step(
            step: TrajectoryStep,
            step_index: int,
            trajectory_messages: list[dict],
            trajectory_id: str,
            gt_tool_call: tuple[str, ...] | None = None,
        ) -> list[dict]:
            del trajectory_messages, trajectory_id
            nonlocal active_workers, max_active_workers
            async with lock:
                active_workers += 1
                max_active_workers = max(max_active_workers, active_workers)
            await asyncio.sleep(0.01)
            async with lock:
                active_workers -= 1
            return [
                {
                    "step_index": step_index,
                    "step_id": step.message_index,
                    "ground_truth": list(gt_tool_call or ()),
                }
            ]

        pipeline.process_step = fake_process_step
        with patch.object(data_preparation, "parse_trajectory", return_value=trajectory):
            examples = await pipeline.process_trajectory(trajectory_path)

        assert max_active_workers == 2
        assert [example["step_id"] for example in examples] == [1, 2, 3]
        assert examples[0]["ground_truth"] == ["grep two app.py"]
        assert examples[1]["ground_truth"] == ["grep three app.py"]
        assert examples[2]["ground_truth"] == []

    @pytest.mark.asyncio
    async def test_generate_compressions_uses_rollout_api_key(self, config: Config) -> None:
        """SFT rollout completions should use the rollout-specific API key."""
        rollout_config = replace(
            config.sft.rollout,
            api_endpoint="http://rollout.example/v1",
            api_key="rollout-key",
        )
        sft_config = replace(config.sft, rollout=rollout_config)
        config_with_key = replace(config, sft=sft_config)
        compression_filter = CompressionFilter(
            config.sft.data_preparation.skip_compression_max_tokens,
            _FakeTokenizer(),
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="compressed"))]
        )

        with patch.object(
            data_preparation, "acompletion", AsyncMock(return_value=response)
        ) as mock:
            with patch.object(data_preparation, "MiniSWERunner", return_value=MagicMock()):
                with patch.object(
                    data_preparation.CompressionFilter,
                    "build",
                    return_value=compression_filter,
                ):
                    pipeline = data_preparation.DataPreparationPipeline(config_with_key)
                    outputs = await pipeline.generate_compressions("prompt", num_samples=1)

        assert outputs == ["compressed"]
        completion_kwargs = mock.await_args.kwargs
        assert completion_kwargs["api_base"] == rollout_config.api_endpoint
        assert completion_kwargs["api_key"] == "rollout-key"

    @pytest.mark.asyncio
    async def test_process_step_does_not_skip_code_view_pipeline(self, pipeline) -> None:
        """Test that `cat | head` no longer bypasses compression."""
        long_output = " ".join(f"token{i}" for i in range(600))
        step = _make_step("cat app.py | head -20", long_output)
        pipeline.generate_compressions = AsyncMock(return_value=["compressed output"])
        pipeline._infer_sample_action = AsyncMock(
            return_value=_make_rollout_sample(
                compressed_text="compressed output",
                keep_original=False,
                inferred_action="grep pattern app.py",
                trajectory_id="traj-001",
            )
        )

        with patch.object(
            data_preparation,
            "sample_anchor_set",
            AsyncMock(return_value=["grep pattern app.py"]),
        ):
            examples = await pipeline.process_step(
                step=step,
                step_index=0,
                trajectory_messages=[],
                trajectory_id="traj-001",
                gt_tool_call=("grep pattern app.py",),
            )

        assert len(examples) == 1
        assert examples[0]["completion"] == "compressed output"
        # inferred action matches the single anchor exactly -> reward 1.0
        assert examples[0]["action_reward"] == pytest.approx(1.0)
        assert examples[0]["length_reward"] == pytest.approx(0.0)
        assert examples[0]["anchor_size"] == 1
        assert examples[0]["anchor_aggregator"] == "top3"
        assert "total_reward" not in examples[0]
        pipeline.generate_compressions.assert_awaited_once()
        assert len(pipeline.filter_counts) == 0

    @pytest.mark.asyncio
    async def test_process_step_returns_all_valid_rollout_samples(self, pipeline) -> None:
        """Test that `process_step` preserves all raw rollout samples."""
        long_output = " ".join(f"token{i}" for i in range(600))
        step = _make_step("grep TODO app.py", long_output)
        pipeline.generate_compressions = AsyncMock(
            return_value=["brief", "longer compression text"]
        )
        pipeline._infer_sample_action = AsyncMock(
            side_effect=[
                _make_rollout_sample(
                    compressed_text="brief",
                    keep_original=False,
                    inferred_action="grep pattern app.py",
                    trajectory_id="traj-002",
                ),
                _make_rollout_sample(
                    compressed_text="longer compression text",
                    keep_original=False,
                    inferred_action="grep pattern app.py",
                    trajectory_id="traj-002",
                ),
            ]
        )

        with patch.object(
            data_preparation,
            "sample_anchor_set",
            AsyncMock(return_value=["grep pattern app.py"]),
        ):
            examples = await pipeline.process_step(
                step=step,
                step_index=0,
                trajectory_messages=[],
                trajectory_id="traj-002",
                gt_tool_call=("grep pattern app.py",),
            )

        assert len(examples) == 2
        example_by_completion = {example["completion"]: example for example in examples}
        assert example_by_completion["brief"]["action_reward"] == pytest.approx(1.0)
        assert example_by_completion["longer compression text"]["action_reward"] == pytest.approx(
            1.0
        )
        # Within-group length reward: "brief" (shortest) gets +0.5, the longer
        # one gets -0.5 (both pass the action gate so both are length-scored).
        assert example_by_completion["brief"]["length_reward"] == pytest.approx(0.5)
        assert example_by_completion["longer compression text"]["length_reward"] == pytest.approx(
            -0.5
        )
        assert all("total_reward" not in example for example in examples)
        assert all("score" not in example for example in examples)

    @pytest.mark.asyncio
    async def test_process_step_saves_raw_model_outputs(self, pipeline, tmp_path: Path) -> None:
        """Raw compressor outputs should be written to a sidecar JSONL file."""
        long_output = " ".join(f"token{i}" for i in range(600))
        step = _make_step("grep TODO app.py", long_output)
        pipeline.raw_model_output_file = tmp_path / "raw_model_outputs.jsonl"
        raw_output_one = '{"type":"plain","content":"brief"}'
        raw_output_two = '{"type":"plain","content":"longer compression text"}'
        pipeline.generate_compressions = AsyncMock(return_value=[raw_output_one, raw_output_two])
        pipeline._infer_sample_action = AsyncMock(
            side_effect=[
                _make_rollout_sample(
                    compressed_text=raw_output_one,
                    keep_original=False,
                    inferred_action="grep pattern app.py",
                    trajectory_id="traj-raw",
                    normalized_completion=raw_output_one,
                ),
                _make_rollout_sample(
                    compressed_text=raw_output_two,
                    keep_original=False,
                    inferred_action="grep pattern app.py",
                    trajectory_id="traj-raw",
                    normalized_completion=raw_output_two,
                ),
            ]
        )

        with patch.object(
            data_preparation,
            "sample_anchor_set",
            AsyncMock(return_value=["grep pattern app.py"]),
        ):
            await pipeline.process_step(
                step=step,
                step_index=0,
                trajectory_messages=[],
                trajectory_id="traj-raw",
                gt_tool_call=("grep pattern app.py",),
            )

        records = _load_jsonl(pipeline.raw_model_output_file)
        assert len(records) == 2
        assert records[0]["trajectory_id"] == "traj-raw"
        assert records[0]["step_id"] == step.message_index
        assert records[0]["sample_index"] == 0
        assert records[0]["raw_completion"] == raw_output_one
        assert records[0]["prompt"].startswith("You are a context compression assistant.")
        assert "## Context Focus Question" in records[0]["prompt"]
        assert records[1]["sample_index"] == 1
        assert records[1]["raw_completion"] == raw_output_two

    @pytest.mark.asyncio
    async def test_process_step_keeps_explicit_unchanged_samples(self, pipeline) -> None:
        """Explicit unchanged JSON samples should remain in SFT data."""
        original_output = "very long output " * 200
        step = _make_step("grep TODO app.py", original_output)
        unchanged_completion = '{"type":"unchanged","content":null}'
        plain_completion = '{"type":"plain","content":"short summary"}'
        pipeline.generate_compressions = AsyncMock(
            return_value=[unchanged_completion, plain_completion]
        )
        pipeline._infer_sample_action = AsyncMock(
            side_effect=[
                _make_rollout_sample(
                    compressed_text=unchanged_completion,
                    keep_original=True,
                    inferred_action="grep pattern app.py",
                    trajectory_id="traj-keep",
                    normalized_completion=unchanged_completion,
                ),
                _make_rollout_sample(
                    compressed_text=plain_completion,
                    keep_original=False,
                    inferred_action="grep pattern app.py",
                    trajectory_id="traj-keep",
                    normalized_completion=plain_completion,
                ),
            ]
        )

        with patch.object(
            data_preparation,
            "sample_anchor_set",
            AsyncMock(return_value=["grep pattern app.py"]),
        ):
            examples = await pipeline.process_step(
                step=step,
                step_index=0,
                trajectory_messages=[],
                trajectory_id="traj-keep",
                gt_tool_call=("grep pattern app.py",),
            )

        assert len(examples) == 2
        example_by_completion = {example["completion"]: example for example in examples}
        assert example_by_completion[unchanged_completion]["action_reward"] == pytest.approx(1.0)
        assert example_by_completion[plain_completion]["action_reward"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_process_step_drops_step_when_anchor_set_empty(self, pipeline) -> None:
        """A step is dropped (fail-fast) when no anchor samples are produced."""
        long_output = " ".join(f"token{i}" for i in range(600))
        step = _make_step("grep TODO app.py", long_output)
        pipeline.generate_compressions = AsyncMock(return_value=["compressed output"])
        pipeline._infer_sample_action = AsyncMock(
            return_value=_make_rollout_sample(
                compressed_text="compressed output",
                keep_original=False,
                inferred_action="grep pattern app.py",
                trajectory_id="traj-anchorless",
            )
        )

        with patch.object(data_preparation, "sample_anchor_set", AsyncMock(return_value=[])):
            examples = await pipeline.process_step(
                step=step,
                step_index=0,
                trajectory_messages=[],
                trajectory_id="traj-anchorless",
                gt_tool_call=("grep pattern app.py",),
            )

        assert examples == []

    @pytest.mark.asyncio
    async def test_process_step_returns_empty_when_all_samples_are_invalid_fallback(
        self,
        pipeline,
    ) -> None:
        """A step should be dropped if every rollout sample resolves to invalid fallback."""
        original_output = "very long output " * 200
        step = _make_step("grep TODO app.py", original_output)
        invalid_completion = '{"type":"code","content":["2-"]}'
        pipeline.generate_compressions = AsyncMock(return_value=[invalid_completion])
        pipeline._infer_sample_action = AsyncMock(
            return_value=_make_rollout_sample(
                compressed_text=invalid_completion,
                keep_original=True,
                inferred_action="grep pattern app.py",
                trajectory_id="traj-keep-only",
                normalized_completion=None,
                valid_response=False,
            )
        )

        examples = await pipeline.process_step(
            step=step,
            step_index=0,
            trajectory_messages=[],
            trajectory_id="traj-keep-only",
            gt_tool_call=("grep pattern app.py",),
        )

        assert examples == []

    @pytest.mark.asyncio
    async def test_process_step_skips_short_output(self, pipeline) -> None:
        """Test that short outputs are skipped due to token count."""
        step = _make_step("grep TODO app.py", "small output")
        pipeline.generate_compressions = AsyncMock()

        examples = await pipeline.process_step(
            step=step,
            step_index=0,
            trajectory_messages=[],
            trajectory_id="traj-003",
            gt_tool_call=("grep pattern app.py",),
        )

        assert examples == []
        pipeline.generate_compressions.assert_not_awaited()
        assert pipeline.filter_counts[CompressionFilterReason.SHORT_OUTPUT] == 1

    @pytest.mark.asyncio
    async def test_process_step_skips_missing_context_focus_question(self, pipeline) -> None:
        """Steps without focus questions should bypass semantic compression."""
        step = TrajectoryStep(
            goal="Fix a bug",
            message_index=1,
            tool_output=(ToolOutput(returncode=0, output="long output " * 200),),
            reasoning="Inspect the repository state",
            tool_call=("grep TODO app.py",),
            context_focus_question=(None,),
        )
        pipeline.generate_compressions = AsyncMock()

        examples = await pipeline.process_step(
            step=step,
            step_index=0,
            trajectory_messages=[],
            trajectory_id="traj-no-focus",
            gt_tool_call=("grep pattern app.py",),
        )

        assert examples == []
        pipeline.generate_compressions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_step_skips_multi_tool_output_steps(self, pipeline) -> None:
        """Off-policy rollout data only supports single tool-call compression targets."""
        step = TrajectoryStep(
            goal="Fix a bug",
            message_index=1,
            tool_output=(
                ToolOutput(returncode=0, output="first long output " * 200),
                ToolOutput(returncode=0, output="second long output " * 200),
            ),
            reasoning="Inspect repository state",
            tool_call=("ls -la", "git status"),
            context_focus_question=(
                "List files in the repository root.",
                "Check whether the worktree is clean.",
            ),
        )
        pipeline.generate_compressions = AsyncMock()

        examples = await pipeline.process_step(
            step=step,
            step_index=0,
            trajectory_messages=[],
            trajectory_id="traj-multi-tool",
            gt_tool_call=("grep pattern app.py",),
        )

        assert examples == []
        pipeline.generate_compressions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_step_skips_unparseable_command(self, pipeline) -> None:
        """Test that unparseable bash commands result in skip."""
        step = _make_step("python -c 'print(1)'", "long enough output " * 200)
        pipeline.generate_compressions = AsyncMock()

        with patch(
            "src.compression.compression_filter.parse_bash_command",
            side_effect=BashParseError("failed to parse"),
        ):
            examples = await pipeline.process_step(
                step=step,
                step_index=0,
                trajectory_messages=[],
                trajectory_id="traj-004",
                gt_tool_call=("grep pattern app.py",),
            )

        assert examples == []
        pipeline.generate_compressions.assert_not_awaited()
        assert pipeline.filter_counts[CompressionFilterReason.UNPARSEABLE_COMMAND] == 1

    @pytest.mark.asyncio
    async def test_infer_sample_action_uses_original_output_for_continue(self, pipeline) -> None:
        """Unchanged JSON should continue from the untouched tool message."""
        step = _make_step("grep TODO app.py", "raw tool output")
        trajectory_messages = [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": "inspect file"},
            {"role": "tool", "content": "original tool message content"},
        ]
        pipeline.infer_action_with_compression = AsyncMock(return_value="grep pattern app.py")

        sample = await pipeline._infer_sample_action(
            index=0,
            compressed_text='{"type":"unchanged","content":null}',
            step=step,
            trajectory_messages=trajectory_messages,
            trajectory_id="traj-keep",
        )

        assert sample is not None
        assert sample.compressed_text == '{"type":"unchanged","content":null}'
        assert sample.keep_original is True
        assert sample.valid_response is True
        assert sample.normalized_completion == '{"type":"unchanged","content":null}'
        pipeline.infer_action_with_compression.assert_awaited_once()
        call_kwargs = pipeline.infer_action_with_compression.await_args.kwargs
        assert call_kwargs["compressed_output"] == "1> raw tool output"
        assert call_kwargs["use_original_output"] is True

    @pytest.mark.asyncio
    async def test_infer_sample_action_reconstructs_valid_citations(self, pipeline) -> None:
        """Valid citations should inject reconstructed original lines."""
        step = _make_step("grep TODO app.py", "alpha\nbeta\ngamma")
        trajectory_messages = [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": "inspect file"},
            {"role": "tool", "content": "original tool message content"},
        ]
        pipeline.infer_action_with_compression = AsyncMock(return_value="grep pattern app.py")

        sample = await pipeline._infer_sample_action(
            index=0,
            compressed_text='{"type":"code","content":["2-3:remaining lines"]}',
            step=step,
            trajectory_messages=trajectory_messages,
            trajectory_id="traj-citation",
        )

        assert sample is not None
        assert sample.keep_original is False
        call_kwargs = pipeline.infer_action_with_compression.await_args.kwargs
        assert call_kwargs["compressed_output"] == "\n".join(
            [
                "alpha",
                "(compressed 2 lines: remaining lines)",
            ]
        )
        assert call_kwargs["use_original_output"] is False

    @pytest.mark.asyncio
    async def test_infer_sample_action_keeps_non_adjacent_code_ranges_separate(
        self,
        pipeline,
    ) -> None:
        """Non-adjacent omit ranges should stay separate without merging."""
        step = _make_step("grep TODO app.py", "alpha\nbeta\ngamma")
        trajectory_messages = [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": "inspect file"},
            {"role": "tool", "content": "original tool message content"},
        ]
        pipeline.infer_action_with_compression = AsyncMock(return_value="grep pattern app.py")

        sample = await pipeline._infer_sample_action(
            index=0,
            compressed_text='{"type":"code","content":["1:first line","3:third line"]}',
            step=step,
            trajectory_messages=trajectory_messages,
            trajectory_id="traj-merged-citation",
        )

        assert sample is not None
        assert sample.keep_original is False
        assert sample.valid_response is True
        assert (
            sample.normalized_completion
            == '{"type":"code","content":["1:first line","3:third line"]}'
        )
        call_kwargs = pipeline.infer_action_with_compression.await_args.kwargs
        assert call_kwargs["compressed_output"] == "\n".join(
            [
                "(compressed 1 lines: first line)",
                "beta",
                "(compressed 1 lines: third line)",
            ]
        )
        assert call_kwargs["use_original_output"] is False

    @pytest.mark.asyncio
    async def test_infer_sample_action_invalid_citation_falls_back_to_original(
        self, pipeline
    ) -> None:
        """Invalid citations should be treated as original-output fallback."""
        step = _make_step("grep TODO app.py", "alpha\nbeta\ngamma")
        trajectory_messages = [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": "inspect file"},
            {"role": "tool", "content": "original tool message content"},
        ]
        pipeline.infer_action_with_compression = AsyncMock(return_value="grep pattern app.py")

        sample = await pipeline._infer_sample_action(
            index=0,
            compressed_text='{"type":"code","content":["2-"]}',
            step=step,
            trajectory_messages=trajectory_messages,
            trajectory_id="traj-invalid-citation",
        )

        assert sample is not None
        assert sample.keep_original is True
        assert sample.valid_response is False
        assert sample.normalized_completion is None
        call_kwargs = pipeline.infer_action_with_compression.await_args.kwargs
        assert call_kwargs["compressed_output"] == "\n".join(
            [
                "1> alpha",
                "2> beta",
                "3> gamma",
            ]
        )
        assert call_kwargs["use_original_output"] is True
