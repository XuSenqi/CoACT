#!/usr/bin/env python3
"""Evaluate compression strategies on SWE-bench Verified.

Usage:
    # Vanilla baseline
    python scripts/evaluate.py --strategy vanilla --max-instances 50

    # Sliding window baseline
    python scripts/evaluate.py --strategy sliding_window --max-instances 50

    # CoACT compression
    python scripts/evaluate.py --strategy CoACT --max-instances 50

    # Sliding Window + CoACT
    python scripts/evaluate.py --strategy sliding_window_CoACT --max-instances 50

    # AgentDiet + CoACT
    python scripts/evaluate.py --strategy agentdiet_CoACT --max-instances 50

    # LLMLingua-2 (requires the service: scripts/vllm/serve_llmlingua2.sh)
    python scripts/evaluate.py --strategy llmlingua2 --max-instances 50

    # LongCodeZip (requires the service: scripts/vllm/serve_longcodezip.sh)
    python scripts/evaluate.py --strategy longcodezip --max-instances 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# import weave  # noqa: F401
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.config import Config
from src.eval.baselines import get_strategy
from src.eval.metrics import AggregatedMetrics
from src.eval.run_management import (
    EvalWandbTracker,
    build_eval_run_id,
    initialize_manifest,
    prepare_eval_run,
    save_json,
    update_strategy_manifest,
    update_wandb_manifest,
    write_config_snapshot,
)
from src.eval.runner import EvaluationRunner, load_swebench_verified
from src.eval.swebench_harness import SwebenchHarnessResult, run_swebench_harness
from src.utils.cost_tracker import set_active_pricing

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
console = Console()


def load_summary_for_display(
    summary_path: str | Path,
) -> dict[str, Any]:
    """Load a saved summary.json and convert it for rich display."""
    summary_path = Path(summary_path)
    with summary_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    aggregated = AggregatedMetrics(
        method=payload["method"],
        total_instances=payload["total_instances"],
        successful_submissions=payload["successful_submissions"],
        failed_submissions=payload["failed_submissions"],
        avg_prompt_tokens=payload["avg_prompt_tokens"],
        avg_completion_tokens=payload["avg_completion_tokens"],
        avg_total_tokens=payload["avg_total_tokens"],
        total_prompt_tokens=payload["total_prompt_tokens"],
        total_completion_tokens=payload["total_completion_tokens"],
        avg_api_calls=payload["avg_api_calls"],
        total_api_calls=payload["total_api_calls"],
        avg_compression_time_ms=payload["avg_compression_time_ms"],
        total_compression_time_ms=payload["total_compression_time_ms"],
        avg_compressed_steps=payload.get("avg_compressed_steps", 0.0),
        total_compressed_steps=payload.get("total_compressed_steps", 0),
        trajectory_generation_time_seconds=payload.get("trajectory_generation_time_seconds", 0.0),
        timestamp=payload.get("timestamp", ""),
    )

    swebench_payload = payload.get("swebench_harness")
    swebench_result = (
        SwebenchHarnessResult(
            report_path=swebench_payload["report_path"],
            predictions_path=swebench_payload["predictions_path"],
            total_instances=swebench_payload["total_instances"],
            submitted_instances=swebench_payload["submitted_instances"],
            completed_instances=swebench_payload["completed_instances"],
            resolved_instances=swebench_payload["resolved_instances"],
            unresolved_instances=swebench_payload["unresolved_instances"],
            empty_patch_instances=swebench_payload["empty_patch_instances"],
            error_instances=swebench_payload["error_instances"],
        )
        if isinstance(swebench_payload, dict)
        else None
    )

    strategy_name = payload.get("strategy") or payload.get("method", "unknown")
    run_id = payload.get("run_id") or summary_path.parents[2].name
    output_dir = Path(
        payload.get("trajectory_dir", summary_path.parents[2] / "trajectories" / strategy_name)
    )
    swebench_error = payload.get("swebench_harness_error")

    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "aggregated": aggregated,
        "swebench_result": swebench_result,
        "swebench_error": swebench_error,
        "results_path": summary_path,
        "output_dir": output_dir,
    }


def filter_existing_instances(
    instances: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Filter out instances that already have trajectory files in output directory.

    Returns:
        Tuple of (remaining_instances, skipped_existing_count).
    """
    existing_instance_ids = {trajectory_path.stem for trajectory_path in output_dir.glob("*.json")}
    if not existing_instance_ids:
        return instances, 0

    remaining_instances = []
    skipped_existing_count = 0

    for instance in instances:
        instance_id = instance.get("instance_id")
        if instance_id and instance_id in existing_instance_ids:
            skipped_existing_count += 1
            continue
        remaining_instances.append(instance)

    return remaining_instances, skipped_existing_count


def build_wandb_init_payload(
    config: Config,
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, Any]:
    """Build the W&B config payload for an evaluation run."""
    return {
        "run_id": run_id,
        "strategy": args.strategy,
        "config_path": args.config,
        "dataset_name": config.data.eval_dataset,
        "split": config.data.eval_split,
        "seed": config.sft.seed,
        "shuffle": args.shuffle,
        "max_instances": args.max_instances,
        "instance_ids": args.instance_ids or [],
        "trajectory_max_workers": args.trajectory_max_workers,
    }


def build_results_payload(
    run_id: str,
    strategy: str,
    aggregated,
    swebench_result: SwebenchHarnessResult | None,
    swebench_error: str | None,
    summary_path: Path,
    trajectory_dir: Path,
) -> dict[str, Any]:
    """Assemble the persisted summary payload for a strategy run."""
    results_payload = aggregated.to_dict()
    results_payload["run_id"] = run_id
    results_payload["strategy"] = strategy
    results_payload["summary_path"] = str(summary_path)
    results_payload["trajectory_dir"] = str(trajectory_dir)
    if swebench_result is not None:
        results_payload["swebench_harness"] = swebench_result.to_dict()
    if swebench_error is not None:
        results_payload["swebench_harness_error"] = swebench_error
    return results_payload


def extract_wandb_metrics(
    aggregated,
    swebench_result: SwebenchHarnessResult | None,
) -> dict[str, Any]:
    """Flatten evaluation metrics for W&B logging."""
    metrics = aggregated.to_dict()
    if swebench_result is not None:
        metrics.update(
            {
                "pass_at_1": swebench_result.pass_at_1,
                "resolved_instances": swebench_result.resolved_instances,
                "submitted_instances": swebench_result.submitted_instances,
                "completed_instances": swebench_result.completed_instances,
                "empty_patch_instances": swebench_result.empty_patch_instances,
                "error_instances": swebench_result.error_instances,
            }
        )
    return metrics


def print_rich_summary(
    strategy_name: str,
    run_id: str,
    aggregated,
    swebench_result: SwebenchHarnessResult | None,
    swebench_error: str | None,
    results_path: Path,
    output_dir: Path,
) -> None:
    """Print evaluation summary with rich tables."""
    console.print()
    console.print(
        Panel.fit(
            f"Evaluation Results: [bold]{strategy_name}[/bold]  Run: [bold]{run_id}[/bold]",
            border_style="cyan",
        )
    )

    overview = Table(show_header=False, box=None)
    overview.add_column("Metric", style="bold")
    overview.add_column("Value")
    overview.add_row("Run ID", run_id)
    overview.add_row("Total instances", f"{aggregated.total_instances}")
    overview.add_row("Successful submissions", f"{aggregated.successful_submissions}")
    overview.add_row("Failed submissions", f"{aggregated.failed_submissions}")
    overview.add_row(
        "Submission success rate",
        f"{aggregated.successful_submissions / max(aggregated.total_instances, 1):.2%}",
    )
    overview.add_row(
        "Trajectory generation time",
        f"{aggregated.trajectory_generation_time_seconds:.1f}s",
    )
    console.print(Panel(overview, title="Overview", border_style="blue"))

    observed_tokens = Table(show_header=False, box=None)
    observed_tokens.add_column("Metric", style="bold")
    observed_tokens.add_column("Value")
    observed_tokens.add_row("Total prompt tokens", f"{aggregated.total_prompt_tokens:,}")
    observed_tokens.add_row("Total completion tokens", f"{aggregated.total_completion_tokens:,}")
    observed_tokens.add_row("Avg tokens per instance", f"{aggregated.avg_total_tokens:.0f}")
    observed_tokens.add_row("Total API calls", f"{aggregated.total_api_calls}")
    observed_tokens.add_row("Avg API calls per instance", f"{aggregated.avg_api_calls:.1f}")
    console.print(Panel(observed_tokens, title="Observed Tokens", border_style="green"))

    compression = Table(show_header=False, box=None)
    compression.add_column("Metric", style="bold")
    compression.add_column("Value")
    compression.add_row("Total compression time", f"{aggregated.total_compression_time_ms:.0f}ms")
    compression.add_row("Avg compression time", f"{aggregated.avg_compression_time_ms:.0f}ms")
    console.print(Panel(compression, title="Compression Statistics", border_style="magenta"))

    official = Table(show_header=False, box=None)
    official.add_column("Metric", style="bold")
    official.add_column("Value")
    if swebench_result is not None:
        official.add_row("PASS@1", f"{swebench_result.pass_at_1:.2%}")
        official.add_row(
            "Resolved",
            f"{swebench_result.resolved_instances}/{swebench_result.total_instances}",
        )
        official.add_row("Harness report", swebench_result.report_path)
    elif swebench_error is not None:
        official.add_row("Harness evaluation failed", swebench_error)
    else:
        official.add_row("Harness", "Not run")
    console.print(Panel(official, title="Official SWE-bench", border_style="yellow"))

    artifacts = Table(show_header=False, box=None)
    artifacts.add_column("Artifact", style="bold")
    artifacts.add_column("Path")
    artifacts.add_row("Summary", str(results_path))
    artifacts.add_row("Trajectories", str(output_dir))
    console.print(Panel(artifacts, title="Artifacts", border_style="white"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compression strategies on SWE-bench Verified"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "vanilla",
            "sliding_window",
            "sliding_window_CoACT",
            "CoACT",
            "agentdiet",
            "agentdiet_CoACT",
            "swepruner",
            "llmlingua2",
            "longcodezip",
        ],
        help="Compression strategy to use",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of instances to evaluate",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific instance IDs to evaluate",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle the dataset before evaluation",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Evaluation run identifier. A new run ID is generated when omitted.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume writing into an existing run ID.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run instances even when their trajectory files already exist.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Path to an existing summary.json to display with rich output.",
    )
    parser.add_argument(
        "--trajectory-max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for trajectory generation.",
    )
    args = parser.parse_args()

    if args.summary_json:
        print_rich_summary(**load_summary_for_display(args.summary_json))
        return

    if args.strategy is None:
        parser.error("--strategy is required unless --summary-json is provided")
    if args.trajectory_max_workers <= 0:
        parser.error("--trajectory-max-workers must be positive")

    # Load config
    config = Config.load(args.config)
    set_active_pricing(config.pricing)

    # Get strategy
    strategy = get_strategy(args.strategy, config)

    run_id = args.run_id or build_eval_run_id(
        split=config.data.eval_split,
        max_instances=args.max_instances,
        instance_ids=args.instance_ids,
        seed=config.sft.seed,
        shuffle=args.shuffle,
    )
    run_paths = prepare_eval_run(
        base_dir=config.paths.eval_run_dir,
        run_id=run_id,
        resume=args.resume,
    )
    write_config_snapshot(config, run_paths.config_snapshot_path)

    manifest = initialize_manifest(
        paths=run_paths,
        args=vars(args),
        config_path=args.config,
        dataset_name=config.data.eval_dataset,
        split=config.data.eval_split,
        seed=config.sft.seed,
    )

    wandb_tracker = EvalWandbTracker.create(
        config=config,
        run_id=run_id,
        job_type="evaluation",
        init_payload=build_wandb_init_payload(config=config, args=args, run_id=run_id),
    )
    if wandb_tracker.enabled:
        manifest = update_wandb_manifest(run_paths, wandb_tracker.manifest_payload())

    output_dir = run_paths.strategy_trajectory_dir(args.strategy)
    output_dir.mkdir(parents=True, exist_ok=True)
    update_strategy_manifest(
        paths=run_paths,
        strategy=args.strategy,
        values={
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "trajectory_dir": str(output_dir),
            "summary_path": str(run_paths.strategy_summary_path(args.strategy)),
            "swebench_predictions_path": str(run_paths.strategy_predictions_path(args.strategy)),
            "swebench_report_path": str(run_paths.strategy_report_path(args.strategy)),
        },
    )

    try:
        instances = load_swebench_verified(
            config.data.eval_dataset,
            split=config.data.eval_split,
            shuffle=args.shuffle,
            seed=config.sft.seed,
        )

        if args.instance_ids:
            instance_id_set = set(args.instance_ids)
            instances = [inst for inst in instances if inst.get("instance_id") in instance_id_set]
            logger.info(f"Filtered to {len(instances)} specified instances")

        if args.max_instances is not None:
            instances = instances[: args.max_instances]

        requested_instance_ids = {inst["instance_id"] for inst in instances}

        if args.force_rerun:
            skipped_existing_count = 0
            logger.info("--force-rerun set; not filtering instances by existing trajectory files")
        else:
            instances, skipped_existing_count = filter_existing_instances(instances, output_dir)
            if skipped_existing_count:
                logger.info(
                    f"Skipped {skipped_existing_count} instances with existing trajectories in {output_dir}"
                )

        logger.info(f"Loaded {len(instances)} instances from SWE-bench Verified")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Run root: {run_paths.run_root}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Trajectory workers: {args.trajectory_max_workers}")

        runner = EvaluationRunner(
            config=config,
            strategy=strategy,
            output_dir=str(output_dir),
        )

        _, aggregated = runner.run_dataset(
            instances=instances,
            requested_instance_ids=requested_instance_ids,
            trajectory_max_workers=args.trajectory_max_workers,
        )

        swebench_result = None
        swebench_error = None
        harness_instance_ids = [
            pred["instance_id"] for pred in runner.last_predictions if pred.get("instance_id")
        ]
        if harness_instance_ids:
            try:
                run_paths.strategy_swebench_dir(args.strategy).mkdir(parents=True, exist_ok=True)
                swebench_result = run_swebench_harness(
                    predictions=runner.last_predictions,
                    dataset_name=config.data.eval_dataset,
                    split=config.data.eval_split,
                    instance_ids=harness_instance_ids,
                    max_workers=config.evaluation.swebench_max_workers,
                    timeout=config.evaluation.instance_timeout,
                    run_id=f"{run_id}-{args.strategy}",
                    predictions_path=run_paths.strategy_predictions_path(args.strategy),
                    report_path=run_paths.strategy_report_path(args.strategy),
                )
            except Exception as exc:
                swebench_error = str(exc)
                logger.exception("Official SWE-bench harness evaluation failed")

        results_path = run_paths.strategy_summary_path(args.strategy)
        results_payload = build_results_payload(
            run_id=run_id,
            strategy=args.strategy,
            aggregated=aggregated,
            swebench_result=swebench_result,
            swebench_error=swebench_error,
            summary_path=results_path,
            trajectory_dir=output_dir,
        )
        save_json(results_path, results_payload)

        if wandb_tracker.enabled:
            wandb_tracker.log_metrics(
                args.strategy,
                extract_wandb_metrics(
                    aggregated=aggregated,
                    swebench_result=swebench_result,
                ),
            )

        update_strategy_manifest(
            paths=run_paths,
            strategy=args.strategy,
            values={
                "status": "completed" if swebench_error is None else "completed_with_harness_error",
                "completed_at": datetime.now().isoformat(),
                "requested_instances": aggregated.total_instances,
                "skipped_existing_instances": skipped_existing_count,
                "successful_submissions": aggregated.successful_submissions,
                "failed_submissions": aggregated.failed_submissions,
                "summary_path": str(results_path),
                "trajectory_dir": str(output_dir),
                "swebench_predictions_path": (
                    swebench_result.predictions_path if swebench_result is not None else None
                ),
                "swebench_report_path": (
                    swebench_result.report_path if swebench_result is not None else None
                ),
                "swebench_error": swebench_error,
            },
        )

        if wandb_tracker.enabled:
            wandb_tracker.upload_run_dir(
                run_root=run_paths.run_root,
                metadata={
                    "run_id": run_id,
                    "dataset_name": config.data.eval_dataset,
                    "split": config.data.eval_split,
                    "strategy": args.strategy,
                    "created_at": manifest.get("created_at"),
                    "local_run_root": str(run_paths.run_root),
                },
            )

        print_rich_summary(
            strategy_name=args.strategy,
            run_id=run_id,
            aggregated=aggregated,
            swebench_result=swebench_result,
            swebench_error=swebench_error,
            results_path=results_path,
            output_dir=output_dir,
        )
    finally:
        wandb_tracker.finish()


if __name__ == "__main__":
    main()
