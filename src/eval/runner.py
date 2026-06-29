"""Evaluation runner for compression strategies on SWE-bench."""

import dataclasses
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec

from src.agent.miniswe_runner import MiniSWERunner, create_runner_config_from_config
from src.compression.compression_filter import CompressionFilter
from src.config.config import Config
from src.eval.baselines import (
    AgentDietCoACTCompression,
    CoACTCompression,
    CompressionStrategy,
    SlidingWindowCoACTCompression,
)
from src.eval.compressing_agent import COMPRESSION_STATS_DEFAULTS, CompressingAgent
from src.eval.metrics import AggregatedMetrics, MetricsExtractor, TrajectoryMetrics
from src.utils.concurrency import iter_parallel_completed
from src.utils.cost_tracker import set_active_pricing

logger = logging.getLogger(__name__)


class EvaluationRunner(MiniSWERunner):
    """Runner for evaluating compression strategies on SWE-bench.

    Uses CompressingAgent to apply compression during agent execution.
    """

    def __init__(
        self,
        config: Config,
        strategy: CompressionStrategy,
        output_dir: str,
        agent_class: type[CompressingAgent] = CompressingAgent,
    ):
        """Initialize evaluation runner.

        Args:
            config: Unified configuration.
            strategy: Compression strategy to use.
            output_dir: Directory to save evaluation trajectories.
            agent_class: Agent class to instantiate inside mini-swe-agent.
                Defaults to :class:`CompressingAgent`. Override with a
                subclass to plug in extra hooks while keeping the rest of the
                eval loop intact.
        """
        set_active_pricing(config.pricing)
        runner_config = create_runner_config_from_config(config)
        runner_config = dataclasses.replace(
            runner_config,
            cfq_enabled=strategy.consumes_cfq,
            compression_notice_enabled=strategy.announces_compression,
        )
        super().__init__(runner_config)

        self.project_config = config
        self.strategy = strategy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_class = agent_class
        if isinstance(
            strategy,
            (
                CoACTCompression,
                SlidingWindowCoACTCompression,
                AgentDietCoACTCompression,
            ),
        ):
            self.compression_filter: CompressionFilter | None = CompressionFilter.build(
                config.evaluation.coact.skip_compression_max_tokens
            )
        else:
            self.compression_filter = None

        # Compression tracking
        self._compression_stats: dict[str, Any] = {}
        self.last_predictions: list[dict[str, Any]] = []
        self.prediction_model_name = f"{Path(config.agent.model).name}/{self.strategy.name}"

    def run(
        self,
        task: str,
        instance_id: str | None = None,
        image_name: str | None = None,
    ) -> dict[str, Any]:
        """Run the agent with compression applied during execution.

        Args:
            task: The task/issue description to solve
            instance_id: Optional instance identifier
            image_name: Optional Docker image name for the environment

        Returns:
            Dictionary with trajectory data and compression stats
        """
        # Reset compression stats
        self._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)

        # Run with the configured agent class (CompressingAgent by default)
        result = super().run(
            task=task,
            instance_id=instance_id,
            image_name=image_name,
            agent_class=self.agent_class,
            agent_kwargs={
                "strategy": self.strategy,
                "compression_stats": self._compression_stats,
                "compression_filter": self.compression_filter,
            },
        )

        trajectory_data = result.get("trajectory", {})

        # Add compression stats and collection metadata to trajectory info
        trajectory_data["info"]["exit_status"] = result.get(
            "exit_status",
            trajectory_data["info"].get("exit_status"),
        )
        trajectory_data["info"]["submission"] = result.get(
            "submission",
            trajectory_data["info"].get("submission"),
        )
        trajectory_data["info"]["compression_stats"] = self._compression_stats
        trajectory_data["info"]["collection"] = {
            "collected_at": datetime.now().isoformat(),
            "strategy": self.strategy.name,
        }

        return {
            **result,
            "trajectory": trajectory_data,
        }

    def run_dataset(
        self,
        instances: list[dict[str, Any]],
        requested_instance_ids: set[str] | None = None,
        trajectory_max_workers: int = 1,
    ) -> tuple[list[str], AggregatedMetrics]:
        """Run evaluation on multiple instances.

        Args:
            instances: Instances to generate a trajectory for in this run. When
                resuming, this is the requested scope minus instances that
                already have a trajectory file on disk.
            requested_instance_ids: Full set of instance IDs in scope for this
                run. Defaults to the IDs in ``instances``. This fixes the
                aggregation denominator and bounds which preexisting trajectory
                files in ``output_dir`` are folded back in, so leftover
                trajectories from a larger earlier run that shares the same
                output directory are never counted.
            trajectory_max_workers: Maximum number of worker threads used
                for trajectory generation.

        Returns:
            Tuple of (trajectory file paths, aggregated metrics).
        """
        start_time = time.perf_counter()

        new_instance_ids = {inst["instance_id"] for inst in instances}
        if requested_instance_ids is None:
            requested_instance_ids = new_instance_ids
        out_of_scope = new_instance_ids - requested_instance_ids
        if out_of_scope:
            raise ValueError(
                "run_dataset received instances outside requested_instance_ids: "
                f"{sorted(out_of_scope)}"
            )

        preexisting_paths = [
            fp
            for fp in self.output_dir.glob("*.json")
            if fp.stem in requested_instance_ids and fp.stem not in new_instance_ids
        ]

        trajectory_paths = []
        metrics_list: list[TrajectoryMetrics] = []
        self.last_predictions = []

        def run_instance(
            instance_with_index: tuple[int, dict[str, Any]],
        ) -> tuple[str, dict[str, Any] | None, str]:
            index, instance = instance_with_index
            instance_id = instance["instance_id"]
            logger.info(f"Progress: {index + 1}/{len(instances)} - {instance_id}")

            try:
                runner = EvaluationRunner(
                    config=self.project_config,
                    strategy=self.strategy,
                    output_dir=str(self.output_dir),
                    agent_class=self.agent_class,
                )
                result = runner.run(
                    task=instance.get("problem_statement", ""),
                    instance_id=instance_id,
                    image_name=instance.get("image_name"),
                )
            except Exception as exc:
                logger.error(f"Failed to run {instance_id}: {exc}")
                return instance_id, None, ""

            trajectory_data = result.get("trajectory", {})
            submission = result.get("submission")
            if not submission:
                submission = trajectory_data.get("info", {}).get("submission", "")

            return instance_id, trajectory_data, submission or ""

        completed_results = iter_parallel_completed(
            [(index, instance) for index, instance in enumerate(instances)],
            run_instance,
            max_workers=trajectory_max_workers,
        )

        for instance_id, trajectory_data, submission in completed_results:
            if trajectory_data is None:
                self.last_predictions.append(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": self.prediction_model_name,
                        "model_patch": "",
                    }
                )
                continue

            path = self.output_dir / f"{instance_id}.json"
            self.save_trajectory(trajectory_data, str(path))
            trajectory_paths.append(str(path))

            metrics = MetricsExtractor.extract(
                trajectory_data, model=self.project_config.agent.model
            )
            metrics_list.append(metrics)
            self.last_predictions.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": self.prediction_model_name,
                    "model_patch": submission,
                }
            )

        elapsed = time.perf_counter() - start_time

        for fp in preexisting_paths:
            try:
                data = json.loads(fp.read_text())
            except Exception as exc:
                logger.warning(f"Skipping preexisting trajectory {fp}: {exc}")
                continue
            metrics_list.append(
                MetricsExtractor.extract(data, model=self.project_config.agent.model)
            )
            preexisting_submission = data.get("info", {}).get("submission") or ""
            self.last_predictions.append(
                {
                    "instance_id": fp.stem,
                    "model_name_or_path": self.prediction_model_name,
                    "model_patch": preexisting_submission,
                }
            )

        # Aggregate metrics
        aggregated = MetricsExtractor.aggregate(
            metrics_list,
            method=self.strategy.name,
            trajectory_generation_time_seconds=elapsed,
            total_instances=len(requested_instance_ids),
        )

        return trajectory_paths, aggregated


def load_swebench_verified(
    data_path: str,
    split: str = "test",
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load SWE-bench Verified dataset.

    Args:
        data_path: Path to dataset directory or HuggingFace dataset name.
        split: Dataset split to load.
        shuffle: Whether to shuffle the dataset before materializing instances.
        seed: Random seed used for shuffling.

    Returns:
        List of instances with official SWE-bench Verified Docker image names.
    """
    dataset = load_dataset(data_path, split=split)
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    instances = []
    for raw_instance in dataset:
        instance = dict(raw_instance)
        spec = make_test_spec(instance, namespace="swebench")
        instance["image_name"] = spec.instance_image_key
        instances.append(instance)

    return instances
