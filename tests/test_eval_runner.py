"""Tests for evaluation runner aggregation behavior."""

import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from src.eval.baselines import VanillaCompression
from src.eval.runner import EvaluationRunner


def _fake_project_config() -> object:
    """Minimal project_config stub with the fields accessed by run_dataset."""
    return types.SimpleNamespace(agent=types.SimpleNamespace(model="openai/test-model"))


def _trajectory_payload(instance_id: str, submission: str) -> str:
    """Serialize a minimal trajectory file as written to an output directory."""
    return json.dumps(
        {
            "info": {
                "instance_id": instance_id,
                "exit_status": "Submitted",
                "submission": submission,
                "model_stats": {"api_calls": 1},
            },
            "messages": [],
        }
    )


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    def test_run_dataset_counts_failed_instances(self, tmp_path: Path, monkeypatch) -> None:
        """Exceptions during a run should still count as failed instances."""
        import src.eval.runner as runner_module

        original_runner_class = runner_module.EvaluationRunner

        class FakeWorkerRunner:
            def __init__(self, *, config, strategy, output_dir: str, agent_class=None) -> None:
                self.output_dir = Path(output_dir)
                self.agent_class = agent_class

            def run(
                self,
                task: str,
                instance_id: str | None = None,
                image_name: str | None = None,
            ) -> dict:
                if instance_id == "broken":
                    raise RuntimeError("boom")

                return {
                    "trajectory": {
                        "info": {
                            "instance_id": instance_id,
                            "exit_status": "Submitted",
                            "submission": "diff --git a/file.py b/file.py",
                            "model_stats": {"api_calls": 2},
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "Checking the file",
                                "extra": {
                                    "response": {
                                        "usage": {
                                            "prompt_tokens": 100,
                                            "completion_tokens": 20,
                                        }
                                    }
                                },
                            }
                        ],
                    }
                }

        monkeypatch.setattr(runner_module, "EvaluationRunner", FakeWorkerRunner)

        runner = object.__new__(original_runner_class)
        runner.project_config = _fake_project_config()
        runner.strategy = VanillaCompression()
        runner.output_dir = tmp_path
        runner.prediction_model_name = "Qwen3.5-35B-A3B-FP8/vanilla"
        runner.save_trajectory = lambda trajectory_data, path: None
        runner.agent_class = None

        _, aggregated = original_runner_class.run_dataset(
            runner,
            instances=[
                {"instance_id": "good", "problem_statement": "task a"},
                {"instance_id": "broken", "problem_statement": "task b"},
            ],
        )

        assert aggregated.total_instances == 2
        assert aggregated.successful_submissions == 1
        assert aggregated.failed_submissions == 1
        assert aggregated.total_api_calls == 2
        assert aggregated.avg_prompt_tokens == 100.0
        assert aggregated.avg_api_calls == 2.0
        assert runner.last_predictions == [
            {
                "instance_id": "good",
                "model_name_or_path": "Qwen3.5-35B-A3B-FP8/vanilla",
                "model_patch": "diff --git a/file.py b/file.py",
            },
            {
                "instance_id": "broken",
                "model_name_or_path": "Qwen3.5-35B-A3B-FP8/vanilla",
                "model_patch": "",
            },
        ]

    def test_run_dataset_parallel_records_failures_as_empty_patches(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Failed worker runs should still emit empty patches."""
        import src.eval.runner as runner_module

        original_runner_class = runner_module.EvaluationRunner

        class FakeWorkerRunner:
            def __init__(self, *, config, strategy, output_dir: str, agent_class=None) -> None:
                self.output_dir = Path(output_dir)
                self.agent_class = agent_class

            def run(
                self,
                task: str,
                instance_id: str | None = None,
                image_name: str | None = None,
            ) -> dict:
                if instance_id == "broken":
                    raise RuntimeError("boom")

                return {
                    "trajectory": {
                        "info": {
                            "instance_id": instance_id,
                            "exit_status": "Submitted",
                            "submission": f"patch-{instance_id}",
                            "model_stats": {"api_calls": 1},
                        },
                        "messages": [],
                    },
                    "submission": f"patch-{instance_id}",
                }

        monkeypatch.setattr(runner_module, "EvaluationRunner", FakeWorkerRunner)

        runner = object.__new__(original_runner_class)
        runner.project_config = _fake_project_config()
        runner.strategy = VanillaCompression()
        runner.output_dir = tmp_path
        runner.prediction_model_name = "Qwen3.5-35B-A3B-FP8/vanilla"
        runner.save_trajectory = lambda trajectory_data, path: None
        runner.agent_class = None

        _, aggregated = original_runner_class.run_dataset(
            runner,
            instances=[
                {"instance_id": "good", "problem_statement": "task a"},
                {"instance_id": "broken", "problem_statement": "task b"},
            ],
            trajectory_max_workers=2,
        )

        assert aggregated.total_instances == 2
        assert aggregated.successful_submissions == 1
        assert aggregated.failed_submissions == 1
        assert runner.last_predictions == [
            {
                "instance_id": "good",
                "model_name_or_path": "Qwen3.5-35B-A3B-FP8/vanilla",
                "model_patch": "patch-good",
            },
            {
                "instance_id": "broken",
                "model_name_or_path": "Qwen3.5-35B-A3B-FP8/vanilla",
                "model_patch": "",
            },
        ]

    def test_run_dataset_ignores_preexisting_outside_requested_scope(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Leftover trajectories outside requested_instance_ids must not be counted.

        Reproduces the resume bug: re-running with a smaller --max-instances into a
        directory that still holds trajectories from a larger earlier run must report
        only the requested scope, never the leftover out-of-scope trajectories.
        """
        import src.eval.runner as runner_module

        original_runner_class = runner_module.EvaluationRunner

        class FakeWorkerRunner:
            def __init__(self, *, config, strategy, output_dir: str, agent_class=None) -> None:
                self.output_dir = Path(output_dir)
                self.agent_class = agent_class

            def run(
                self,
                task: str,
                instance_id: str | None = None,
                image_name: str | None = None,
            ) -> dict:
                return {
                    "trajectory": {
                        "info": {
                            "instance_id": instance_id,
                            "exit_status": "Submitted",
                            "submission": f"patch-{instance_id}",
                            "model_stats": {"api_calls": 1},
                        },
                        "messages": [],
                    },
                    "submission": f"patch-{instance_id}",
                }

        monkeypatch.setattr(runner_module, "EvaluationRunner", FakeWorkerRunner)

        # One preexisting trajectory inside the requested scope (already done) and one
        # leftover trajectory outside it (from a larger earlier run sharing the dir).
        (tmp_path / "in_scope_done.json").write_text(
            _trajectory_payload("in_scope_done", "patch-done"), encoding="utf-8"
        )
        (tmp_path / "leftover_extra.json").write_text(
            _trajectory_payload("leftover_extra", "patch-extra"), encoding="utf-8"
        )

        runner = object.__new__(original_runner_class)
        runner.project_config = _fake_project_config()
        runner.strategy = VanillaCompression()
        runner.output_dir = tmp_path
        runner.prediction_model_name = "model/vanilla"
        runner.save_trajectory = lambda trajectory_data, path: None
        runner.agent_class = None

        # leftover_extra is deliberately excluded from the requested scope.
        _, aggregated = original_runner_class.run_dataset(
            runner,
            instances=[{"instance_id": "to_run", "problem_statement": "task"}],
            requested_instance_ids={"to_run", "in_scope_done"},
        )

        assert aggregated.total_instances == 2  # not 3 - leftover_extra excluded
        predicted_ids = {pred["instance_id"] for pred in runner.last_predictions}
        assert predicted_ids == {"to_run", "in_scope_done"}
        assert "leftover_extra" not in predicted_ids

    def test_run_dataset_rejects_instances_outside_requested_scope(self, tmp_path: Path) -> None:
        """Running an instance that is not part of the requested scope must fail fast."""
        runner = object.__new__(EvaluationRunner)
        runner.project_config = _fake_project_config()
        runner.strategy = VanillaCompression()
        runner.output_dir = tmp_path
        runner.prediction_model_name = "model/vanilla"
        runner.save_trajectory = lambda trajectory_data, path: None
        runner.agent_class = None

        with pytest.raises(ValueError, match="outside requested_instance_ids"):
            EvaluationRunner.run_dataset(
                runner,
                instances=[{"instance_id": "rogue", "problem_statement": "task"}],
                requested_instance_ids={"expected"},
            )

    def test_run_dataset_rejects_non_positive_worker_count(self, tmp_path: Path) -> None:
        """Trajectory worker counts must be positive."""
        runner = object.__new__(EvaluationRunner)
        runner.project_config = _fake_project_config()
        runner.strategy = VanillaCompression()
        runner.output_dir = tmp_path
        runner.prediction_model_name = "Qwen3.5-35B-A3B-FP8/vanilla"
        runner.save_trajectory = lambda trajectory_data, path: None
        runner.agent_class = None

        with pytest.raises(ValueError, match="max_workers must be positive"):
            EvaluationRunner.run_dataset(
                runner,
                instances=[{"instance_id": "good", "problem_statement": "task"}],
                trajectory_max_workers=0,
            )

    def test_runner_disables_cfq_for_strategies_that_do_not_consume_it(
        self, tmp_path: Path
    ) -> None:
        """cfq_enabled / compression_notice_enabled must follow the strategy flags."""
        from src.config.config import Config
        from src.eval.baselines import (
            AgentDietCoACTCompression,
            AgentDietCompression,
            CoACTCompression,
            LLMLingua2Compression,
            SlidingWindow,
            SlidingWindowCoACTCompression,
            SWEPrunerCompression,
        )

        config = Config.load("config/config.yaml")
        captured: list[tuple[bool, bool]] = []

        def fake_super_init(self, runner_config) -> None:  # noqa: ARG001
            captured.append((runner_config.cfq_enabled, runner_config.compression_notice_enabled))

        with patch("src.eval.runner.MiniSWERunner.__init__", new=fake_super_init):
            for strategy in (
                VanillaCompression(),
                SlidingWindow(window_size=1),
                AgentDietCompression.from_config(config),
                CoACTCompression.from_config(config),
                SlidingWindowCoACTCompression.from_config(config),
                AgentDietCoACTCompression.from_config(config),
                SWEPrunerCompression.from_config(config),
                LLMLingua2Compression.from_config(config),
            ):
                EvaluationRunner(
                    config=config,
                    strategy=strategy,
                    output_dir=str(tmp_path),
                )

        # (cfq_enabled, compression_notice_enabled) per strategy.
        assert captured == [
            (False, False),  # vanilla
            (False, False),  # sliding_window
            (False, False),  # agentdiet
            (True, False),  # CoACT
            (True, False),  # sliding_window_CoACT
            (True, False),  # agentdiet_CoACT
            (True, False),  # swepruner
            (False, True),  # llmlingua2
        ]
