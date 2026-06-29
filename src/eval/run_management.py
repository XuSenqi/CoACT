"""Helpers for evaluation run layouts, manifests, and W&B tracking."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb

from src.config.config import Config

logger = logging.getLogger(__name__)


def build_eval_run_id(
    split: str,
    max_instances: int | None,
    instance_ids: list[str] | None,
    seed: int,
    shuffle: bool,
    now: datetime | None = None,
) -> str:
    """Build a descriptive run identifier for evaluation output."""
    now = now or datetime.now()
    subset_label = "full"
    if instance_ids:
        subset_label = f"ids{len(instance_ids)}"
    elif max_instances is not None:
        subset_label = f"max{max_instances}"

    shuffle_label = "shuf" if shuffle else "ord"
    parts = [
        "eval",
        split.replace("/", "-").replace(" ", "-"),
        subset_label,
        f"s{seed}",
        shuffle_label,
        now.strftime("%Y%m%d-%H%M%S"),
    ]
    return "-".join(parts)


@dataclass(frozen=True)
class EvalRunPaths:
    """Filesystem layout for a single evaluation run."""

    run_id: str
    run_root: Path
    manifest_path: Path
    config_snapshot_path: Path

    @property
    def trajectories_root(self) -> Path:
        return self.run_root / "trajectories"

    @property
    def results_root(self) -> Path:
        return self.run_root / "results"

    def strategy_trajectory_dir(self, strategy: str) -> Path:
        return self.trajectories_root / strategy

    def strategy_results_dir(self, strategy: str) -> Path:
        return self.results_root / strategy

    def strategy_summary_path(self, strategy: str) -> Path:
        return self.strategy_results_dir(strategy) / "summary.json"

    def strategy_swebench_dir(self, strategy: str) -> Path:
        return self.strategy_results_dir(strategy) / "swebench"

    def strategy_predictions_path(self, strategy: str) -> Path:
        return self.strategy_swebench_dir(strategy) / "predictions.json"

    def strategy_report_path(self, strategy: str) -> Path:
        return self.strategy_swebench_dir(strategy) / "report.json"


def prepare_eval_run(
    base_dir: str | Path,
    run_id: str,
    resume: bool,
) -> EvalRunPaths:
    """Create or reopen a run layout."""
    run_root = Path(base_dir) / run_id

    if run_root.exists() and not resume:
        raise FileExistsError(
            f"Evaluation run already exists: {run_root}. "
            "Pass --resume to continue writing to this run."
        )

    run_root.mkdir(parents=True, exist_ok=True)

    return EvalRunPaths(
        run_id=run_id,
        run_root=run_root,
        manifest_path=run_root / "manifest.json",
        config_snapshot_path=run_root / "config.snapshot.yaml",
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load an existing manifest if present."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_config_snapshot(config: Config, path: Path) -> None:
    """Persist the active config used for a run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config.save(path)


def initialize_manifest(
    paths: EvalRunPaths,
    args: dict[str, Any],
    config_path: str,
    dataset_name: str,
    split: str,
    seed: int,
) -> dict[str, Any]:
    """Create the base manifest payload for a run."""
    manifest = load_manifest(paths.manifest_path)
    if manifest:
        return manifest

    manifest = {
        "run_id": paths.run_id,
        "created_at": datetime.now().isoformat(),
        "run_root": str(paths.run_root),
        "config_path": config_path,
        "config_snapshot_path": str(paths.config_snapshot_path),
        "dataset_name": dataset_name,
        "split": split,
        "seed": seed,
        "args": args,
        "strategies": {},
        "wandb": {},
    }
    save_json(paths.manifest_path, manifest)
    return manifest


def update_strategy_manifest(
    paths: EvalRunPaths,
    strategy: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Upsert strategy-specific manifest information."""
    manifest = load_manifest(paths.manifest_path)
    manifest.setdefault("strategies", {})
    existing = manifest["strategies"].get(strategy, {})
    existing.update(values)
    manifest["strategies"][strategy] = existing
    save_json(paths.manifest_path, manifest)
    return manifest


def update_wandb_manifest(paths: EvalRunPaths, values: dict[str, Any]) -> dict[str, Any]:
    """Upsert top-level W&B metadata in the manifest."""
    manifest = load_manifest(paths.manifest_path)
    manifest.setdefault("wandb", {})
    manifest["wandb"].update(values)
    save_json(paths.manifest_path, manifest)
    return manifest


class EvalWandbTracker:
    """Thin wrapper around optional W&B logging for evaluation runs."""

    def __init__(self, *, run: Any = None, wandb_module: Any = None, artifact_name: str = "") -> None:
        self.run = run
        self.wandb = wandb_module
        self.artifact_name = artifact_name

    @property
    def enabled(self) -> bool:
        return self.run is not None and self.wandb is not None

    @classmethod
    def create(
        cls,
        config: Config,
        run_id: str,
        job_type: str,
        init_payload: dict[str, Any],
    ) -> EvalWandbTracker:
        if not config.wandb.enabled or not config.wandb.project:
            return cls()

        os.environ.setdefault("WANDB_PROJECT", config.wandb.project)
        run = wandb.init(
            project=config.wandb.project,
            name=run_id,
            job_type=job_type,
            config=init_payload,
        )
        return cls(run=run, wandb_module=wandb, artifact_name=f"eval-run-{run_id}")

    def manifest_payload(self) -> dict[str, Any]:
        """Return W&B identifiers safe to persist in the manifest."""
        if not self.enabled:
            return {}

        return {
            "project": getattr(self.run, "project", None),
            "run_id": getattr(self.run, "id", None),
            "run_name": getattr(self.run, "name", None),
            "run_url": getattr(self.run, "url", None),
            "artifact_name": self.artifact_name,
        }

    def log_metrics(self, strategy: str, metrics: dict[str, Any]) -> None:
        """Log summary metrics for a strategy."""
        if not self.enabled:
            return

        prefixed = {f"{strategy}/{key}": value for key, value in metrics.items()}
        self.run.log(prefixed)
        for key, value in prefixed.items():
            self.run.summary[key] = value

    def upload_run_dir(
        self,
        run_root: Path,
        metadata: dict[str, Any],
    ) -> None:
        """Upload the entire local run directory as one artifact."""
        if not self.enabled:
            return

        artifact = self.wandb.Artifact(
            self.artifact_name,
            type="evaluation-run",
            metadata=metadata,
        )
        artifact.add_dir(str(run_root))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Close the W&B run if active."""
        if not self.enabled:
            return
        self.run.finish()
