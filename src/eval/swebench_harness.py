"""Helpers for running official SWE-bench harness evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swebench.harness.run_evaluation import main as harness_main


@dataclass(frozen=True)
class SwebenchHarnessResult:
    """Official SWE-bench harness evaluation result."""

    report_path: str
    predictions_path: str
    total_instances: int
    submitted_instances: int
    completed_instances: int
    resolved_instances: int
    unresolved_instances: int
    empty_patch_instances: int
    error_instances: int

    @property
    def pass_at_1(self) -> float:
        """PASS@1 over the evaluated instance set."""
        return self.resolved_instances / max(self.total_instances, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "report_path": self.report_path,
            "predictions_path": self.predictions_path,
            "total_instances": self.total_instances,
            "submitted_instances": self.submitted_instances,
            "completed_instances": self.completed_instances,
            "resolved_instances": self.resolved_instances,
            "unresolved_instances": self.unresolved_instances,
            "empty_patch_instances": self.empty_patch_instances,
            "error_instances": self.error_instances,
            "pass_at_1": self.pass_at_1,
        }


def run_swebench_harness(
    predictions: list[dict[str, Any]],
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    max_workers: int,
    timeout: int,
    run_id: str,
    predictions_path: str | Path,
    report_path: str | Path,
) -> SwebenchHarnessResult:
    """Run official SWE-bench harness and return parsed report stats."""
    predictions_path = Path(predictions_path)
    report_path = Path(report_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    generated_report_path = harness_main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=timeout,
        namespace="swebench",
        rewrite_reports=False,
        modal=False,
        report_dir=str(report_path.parent),
    )

    generated_report_path = Path(generated_report_path)
    if not generated_report_path.is_absolute():
        generated_report_path = Path.cwd() / generated_report_path

    if generated_report_path != report_path:
        generated_report_path.replace(report_path)

    with open(report_path) as f:
        report = json.load(f)

    return SwebenchHarnessResult(
        report_path=str(report_path),
        predictions_path=str(predictions_path),
        total_instances=report.get("total_instances", 0),
        submitted_instances=report.get("submitted_instances", 0),
        completed_instances=report.get("completed_instances", 0),
        resolved_instances=report.get("resolved_instances", 0),
        unresolved_instances=report.get("unresolved_instances", 0),
        empty_patch_instances=report.get("empty_patch_instances", 0),
        error_instances=report.get("error_instances", 0),
    )
