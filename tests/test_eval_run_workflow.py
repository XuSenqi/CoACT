"""Tests for run-based evaluation output management."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts import evaluate
from src.eval.metrics import AggregatedMetrics
from src.eval.run_management import build_eval_run_id, prepare_eval_run
from src.eval.swebench_harness import SwebenchHarnessResult


def write_test_config(path: Path, eval_run_dir: Path) -> None:
    """Write a minimal config suitable for evaluation script tests."""
    path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "model": "openai/Qwen3.5-35B-A3B-FP8",
                    "api_endpoint": "http://localhost:8000/v1",
                },
                "sft": {
                    "model_path": "model/Qwen3.5-4B",
                    "rollout": {
                        "model": "openai/Qwen3.5-4B",
                        "api_endpoint": "http://localhost:8001/v1",
                    },
                },
                "paths": {
                    "eval_run_dir": str(eval_run_dir),
                },
                "wandb": {
                    "project": "CoACT",
                },
            }
        ),
        encoding="utf-8",
    )


@dataclass
class FakeTracker:
    """Fake W&B tracker for testing."""

    enabled: bool = True
    logged_metrics: list[tuple[str, dict]] | None = None
    uploaded_roots: list[Path] | None = None
    finished: bool = False

    def __post_init__(self) -> None:
        self.logged_metrics = []
        self.uploaded_roots = []

    def manifest_payload(self) -> dict[str, str]:
        return {
            "project": "CoACT",
            "run_id": "wandb-run-1",
            "run_name": "eval-test-run",
            "run_url": "https://wandb.example/run",
            "artifact_name": "eval-run-test-run",
        }

    def log_metrics(self, strategy: str, metrics: dict) -> None:
        self.logged_metrics.append((strategy, metrics))

    def upload_run_dir(self, *, run_root: Path, metadata: dict) -> None:
        self.uploaded_roots.append(run_root)

    def finish(self) -> None:
        self.finished = True


def build_aggregated(method: str) -> AggregatedMetrics:
    """Create a representative aggregate payload."""
    return AggregatedMetrics(
        method=method,
        total_instances=1,
        successful_submissions=1,
        failed_submissions=0,
        avg_prompt_tokens=100.0,
        avg_completion_tokens=10.0,
        avg_total_tokens=110.0,
        total_prompt_tokens=100,
        total_completion_tokens=10,
        avg_api_calls=2.0,
        total_api_calls=2,
        avg_compression_time_ms=0.0,
        total_compression_time_ms=0.0,
        avg_compressed_steps=0.0,
        total_compressed_steps=0,
        trajectory_generation_time_seconds=1.5,
        timestamp="2026-03-21T12:00:00",
    )


def test_build_eval_run_id_includes_subset_and_shuffle() -> None:
    run_id = build_eval_run_id(
        split="test",
        max_instances=50,
        instance_ids=None,
        seed=42,
        shuffle=True,
    )

    assert run_id.startswith("eval-test-max50-s42-shuf-")


def test_prepare_eval_run_requires_resume(tmp_path: Path) -> None:
    paths = prepare_eval_run(base_dir=tmp_path, run_id="eval-run-1", resume=False)

    assert paths.run_root.exists()

    try:
        prepare_eval_run(base_dir=tmp_path, run_id="eval-run-1", resume=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected prepare_eval_run to require --resume")


def test_filter_existing_instances_uses_run_scoped_strategy_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval-runs" / "eval-run-1" / "trajectories" / "CoACT"
    output_dir.mkdir(parents=True)
    (output_dir / "repo__1.json").write_text("{}", encoding="utf-8")

    remaining, skipped = evaluate.filter_existing_instances(
        [
            {"instance_id": "repo__1", "problem_statement": "a"},
            {"instance_id": "repo__2", "problem_statement": "b"},
        ],
        output_dir,
    )

    assert skipped == 1
    assert [item["instance_id"] for item in remaining] == ["repo__2"]


def test_evaluate_main_writes_run_outputs_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    eval_run_dir = tmp_path / "eval-runs"
    write_test_config(config_path, eval_run_dir)

    tracker = FakeTracker()
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *, config, strategy, output_dir: str) -> None:
            self.last_predictions = [
                {
                    "instance_id": "repo__1",
                    "model_name_or_path": "model/vanilla",
                    "model_patch": "diff --git a/file.py b/file.py",
                }
            ]
            self.output_dir = Path(output_dir)

        def run_dataset(self, instances, requested_instance_ids=None, trajectory_max_workers=1):
            captured["trajectory_max_workers"] = trajectory_max_workers
            captured["requested_instance_ids"] = requested_instance_ids
            trajectory_path = self.output_dir / "repo__1.json"
            trajectory_path.write_text("{}", encoding="utf-8")
            return [str(trajectory_path)], build_aggregated("vanilla")

    def fake_harness(**kwargs):
        Path(kwargs["predictions_path"]).write_text("[]", encoding="utf-8")
        Path(kwargs["report_path"]).write_text(
            json.dumps({"resolved_instances": 1, "total_instances": 1}),
            encoding="utf-8",
        )
        return SwebenchHarnessResult(
            report_path=str(kwargs["report_path"]),
            predictions_path=str(kwargs["predictions_path"]),
            total_instances=1,
            submitted_instances=1,
            completed_instances=1,
            resolved_instances=1,
            unresolved_instances=0,
            empty_patch_instances=0,
            error_instances=0,
        )

    monkeypatch.setattr(evaluate, "load_swebench_verified", lambda *args, **kwargs: [
        {"instance_id": "repo__1", "problem_statement": "task"}
    ])
    monkeypatch.setattr(evaluate, "EvaluationRunner", FakeRunner)
    monkeypatch.setattr(evaluate, "run_swebench_harness", fake_harness)
    monkeypatch.setattr(
        evaluate,
        "set_active_pricing",
        lambda pricing: captured.update({"pricing_registered": pricing}),
    )

    def fake_tracker_create(cls, **kwargs):
        captured["wandb_init_payload"] = kwargs["init_payload"]
        return tracker

    monkeypatch.setattr(
        evaluate.EvalWandbTracker,
        "create",
        classmethod(fake_tracker_create),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config_path),
            "--strategy",
            "vanilla",
            "--run-id",
            "eval-test-run",
            "--trajectory-max-workers",
            "4",
        ],
    )

    evaluate.main()

    run_root = eval_run_dir / "eval-test-run"
    summary_path = run_root / "results" / "vanilla" / "summary.json"
    manifest_path = run_root / "manifest.json"
    trajectory_path = run_root / "trajectories" / "vanilla" / "repo__1.json"
    report_path = run_root / "results" / "vanilla" / "swebench" / "report.json"
    predictions_path = run_root / "results" / "vanilla" / "swebench" / "predictions.json"

    assert summary_path.exists()
    assert manifest_path.exists()
    assert trajectory_path.exists()
    assert report_path.exists()
    assert predictions_path.exists()
    assert tracker.uploaded_roots == [run_root]
    assert tracker.finished is True
    assert "pricing_registered" in captured

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "eval-test-run"
    assert manifest["strategies"]["vanilla"]["summary_path"] == str(summary_path)
    assert manifest["strategies"]["vanilla"]["trajectory_dir"] == str(run_root / "trajectories" / "vanilla")
    assert manifest["wandb"]["artifact_name"] == "eval-run-test-run"
    assert captured["trajectory_max_workers"] == 4
    assert captured["requested_instance_ids"] == {"repo__1"}
    assert captured["wandb_init_payload"]["trajectory_max_workers"] == 4

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == "eval-test-run"
    assert summary["summary_path"] == str(summary_path)
    assert summary["trajectory_dir"] == str(run_root / "trajectories" / "vanilla")
    assert summary["swebench_harness"]["report_path"] == str(report_path)


def test_evaluate_main_can_render_existing_summary_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "data" / "eval" / "runs" / "eval-test-run" / "results" / "CoACT" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_dir = summary_path.parents[2] / "trajectories" / "CoACT"
    summary_path.write_text(
        json.dumps(
            {
                "method": "CoACT",
                "strategy": "CoACT",
                "run_id": "eval-test-run",
                "total_instances": 2,
                "successful_submissions": 1,
                "failed_submissions": 1,
                "avg_prompt_tokens": 100.0,
                "avg_completion_tokens": 20.0,
                "avg_total_tokens": 120.0,
                "total_prompt_tokens": 200,
                "total_completion_tokens": 40,
                "avg_api_calls": 3.0,
                "total_api_calls": 6,
                "avg_compression_time_ms": 5.0,
                "total_compression_time_ms": 10.0,
                "avg_compressed_steps": 1.0,
                "total_compressed_steps": 2,
                "trajectory_generation_time_seconds": 12.5,
                "timestamp": "2026-03-21T12:00:00",
                "summary_path": str(summary_path),
                "trajectory_dir": str(trajectory_dir),
                "swebench_harness": {
                    "report_path": str(summary_path.parent / "swebench" / "report.json"),
                    "predictions_path": str(summary_path.parent / "swebench" / "predictions.json"),
                    "total_instances": 2,
                    "submitted_instances": 2,
                    "completed_instances": 2,
                    "resolved_instances": 1,
                    "unresolved_instances": 1,
                    "empty_patch_instances": 0,
                    "error_instances": 0,
                    "pass_at_1": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_print_rich_summary(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(evaluate, "print_rich_summary", fake_print_rich_summary)
    monkeypatch.setattr(evaluate.Config, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("config should not load in summary mode")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--summary-json",
            str(summary_path),
        ],
    )

    evaluate.main()

    assert captured["strategy_name"] == "CoACT"
    assert captured["run_id"] == "eval-test-run"
    assert captured["results_path"] == summary_path
    assert captured["output_dir"] == trajectory_dir
    assert captured["aggregated"].total_instances == 2
    assert captured["swebench_result"].resolved_instances == 1
