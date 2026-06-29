#!/usr/bin/env python3
"""Collect agent trajectories from SWE-smith dataset.

This script collects trajectories by running MiniSWERunner on instances
from the SWE-smith dataset. The trajectories are saved for later use in
SFT data preparation.

Usage:
    python scripts/collect_trajectories.py --config config/config.yaml --max-trajectories 50

Features:
    - Loads instances from SWE-smith dataset on HuggingFace
    - Filters instances by available Docker images (optional)
    - Runs MiniSWERunner on each instance
    - Saves trajectories as JSON files
    - Checkpoint/resume support for long-running collections
    - Parallel collection (optional)

Output:
    - Trajectory files in data/trajectories/{instance_id}.json
    - Checkpoint file at data/trajectories/.checkpoint.json
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.miniswe_runner import MiniSWERunner, create_runner_config_from_config
from src.config.config import Config
from src.utils.checkpoint import CheckpointManager
from src.utils.concurrency import iter_parallel_completed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# Docker Image Utilities
# =============================================================================


def get_available_docker_images() -> list[str]:
    """Get list of available SWE-bench Docker images using docker SDK.

    Returns:
        List of normalized Docker image names (without tag).
    """
    try:
        import docker

        client = docker.from_env()
        images = []

        for image in client.images.list():
            for tag in image.tags:
                if tag and ("swebench" in tag or "swesmith" in tag):
                    # Remove tag suffix (e.g., :latest)
                    image_name = tag.split(":")[0] if ":" in tag else tag
                    images.append(image_name)

        logger.info(f"Found {len(images)} SWE-bench Docker images")
        return images

    except ImportError:
        logger.warning("docker package not installed - skipping Docker image filtering")
        return []
    except Exception as e:
        logger.warning(f"Error getting Docker images: {e}")
        return []


def normalize_image_name(image_name: str) -> str:
    """Normalize Docker image name for matching.

    Handles different naming conventions:
    - Dataset format: jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
    - Docker format: swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536:latest

    Args:
        image_name: Raw image name.

    Returns:
        Normalized image name (prefix normalized, tag removed).
    """
    if not image_name:
        return ""

    # Remove tag if present
    name = image_name.split(":")[0] if ":" in image_name else image_name

    # Normalize prefix: various prefixes -> swebench/
    # jyangballin/swesmith -> swebench/swesmith
    name = re.sub(r"^[^/]+/(swesmith\.)", r"swebench/\1", name)

    return name


def filter_instances_by_docker_images(
    instances: list[dict[str, Any]],
    available_images: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter instances to only those with available Docker images.

    Args:
        instances: List of SWE-smith instance dictionaries.
        available_images: List of available Docker image names.
            If None, no filtering is applied.

    Returns:
        Filtered list of instances.
    """
    if available_images is None:
        return instances

    if not available_images:
        logger.warning("No Docker images available - returning empty list")
        return []

    # Normalize available images for matching
    normalized_available = {normalize_image_name(img) for img in available_images}

    filtered = []
    for instance in instances:
        image_name = instance.get("image_name", "")
        normalized = normalize_image_name(image_name)

        if normalized in normalized_available:
            filtered.append(instance)
        else:
            logger.debug(f"Skipping {instance.get('instance_id')}: no Docker image")

    logger.info(
        f"Filtered {len(instances)} instances to {len(filtered)} "
        f"with available Docker images"
    )
    return filtered


# =============================================================================
# Configuration and Classes
# =============================================================================


@dataclass
class CollectionConfig:
    """Configuration for trajectory collection."""

    dataset_name: str
    """HuggingFace dataset name."""

    output_dir: str
    """Directory to save trajectories."""

    max_instances: int | None = None
    """Maximum number of instances to collect (None = all)."""

    start_index: int = 0
    """Start index in dataset (for resuming)."""

    max_workers: int = 1
    """Number of parallel workers (threads)."""

    split: str = "train"
    """Dataset split to use."""

    filter_by_docker: bool = True
    """Filter instances by available Docker images."""

    shuffle: bool = False
    """Randomly shuffle the dataset before collection."""

    seed: int = 42
    """Random seed for shuffling."""


@dataclass
class CollectionStats:
    """Statistics for collection run."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time else None
            ),
        }


class TrajectoryCollector:
    """Collects agent trajectories from SWE-smith dataset."""

    def __init__(
        self,
        collection_config: CollectionConfig,
        config: Config,
    ) -> None:
        """Initialize the collector.

        Args:
            collection_config: Collection-specific configuration.
            config: Unified project configuration.
        """
        self.collection_config = collection_config
        self.config = config

        # Create output directory
        self.output_dir = Path(collection_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize runner config
        self.runner_config = replace(
            create_runner_config_from_config(config),
            trajectory_collection_mode=True,
        )

        # Checkpoint manager
        checkpoint_path = self.output_dir / ".checkpoint.json"
        self.checkpoint = CheckpointManager(checkpoint_path)

    def load_dataset(self) -> Any:
        """Load the SWE-smith dataset from HuggingFace.

        Returns:
            The dataset split (typically 'train').
        """
        logger.info(f"Loading dataset: {self.collection_config.dataset_name}")
        dataset = load_dataset(
            self.collection_config.dataset_name,
            split=self.collection_config.split,
        )
        logger.info(f"Loaded {len(dataset)} instances")

        if self.collection_config.shuffle:
            logger.info(f"Shuffling dataset with seed {self.collection_config.seed}")
            dataset = dataset.shuffle(seed=self.collection_config.seed)

        return dataset

    def get_filtered_dataset(self, dataset: Any) -> list[dict[str, Any]]:
        """Filter dataset by available Docker images.

        Args:
            dataset: The loaded HuggingFace dataset.

        Returns:
            Filtered list of instances.
        """
        if not self.collection_config.filter_by_docker:
            # No filtering - convert to list
            return [dataset[i] for i in range(len(dataset))]

        # Get available Docker images
        available_images = get_available_docker_images()

        if not available_images:
            logger.warning("No Docker images found - filtering disabled")
            if self.collection_config.filter_by_docker:
                logger.warning("Set --no-docker-filter to disable filtering")
                return []

        # Convert dataset to list and filter
        instances = [dataset[i] for i in range(len(dataset))]
        return filter_instances_by_docker_images(instances, available_images)

    def save_trajectory(
        self,
        instance_id: str,
        trajectory_data: dict[str, Any],
    ) -> Path:
        """Save a trajectory to a JSON file.

        Args:
            instance_id: Instance identifier.
            trajectory_data: Trajectory data dictionary.

        Returns:
            Path to saved file.
        """
        # Ensure instance_id is in the trajectory
        if "info" not in trajectory_data:
            trajectory_data["info"] = {}
        trajectory_data["info"]["instance_id"] = instance_id

        # Add collection metadata
        trajectory_data["collection"] = {
            "collected_at": datetime.now().isoformat(),
            "dataset": self.collection_config.dataset_name,
        }

        # Save to file
        output_path = self.output_dir / f"{instance_id}.json"
        with open(output_path, "w") as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved trajectory to {output_path}")
        return output_path

    def collect_instance(self, instance: dict[str, Any]) -> dict[str, Any] | None:
        """Collect a trajectory for a single instance.

        Args:
            instance: SWE-smith instance dictionary.

        Returns:
            Trajectory data or None if failed.
        """
        instance_id = instance.get("instance_id", "unknown")
        image_name = instance.get("image_name")

        try:
            # Format task
            task = instance.get("problem_statement", "")

            # Run agent with Docker image if available
            logger.info(f"Collecting trajectory for {instance_id}")
            if image_name:
                logger.info(f"Using Docker image: {image_name}")
            runner = MiniSWERunner(self.runner_config)
            result = runner.run(
                task,
                instance_id=instance_id,
                image_name=image_name,
            )

            if result is None:
                logger.warning(f"No result for {instance_id}")
                return None

            return result.get("trajectory")

        except Exception as e:
            logger.error(f"Failed to collect {instance_id}: {e}")
            return None

    def collect_all(self) -> dict[str, Any]:
        """Collect trajectories for all instances in dataset.

        Returns:
            Statistics dictionary.
        """
        stats = CollectionStats()

        # Load dataset
        dataset = self.load_dataset()

        # Filter by available Docker images
        instances = self.get_filtered_dataset(dataset)
        if not instances:
            logger.error("No instances available after filtering")
            return stats.to_dict()

        # Load checkpoint
        completed_ids = self.checkpoint.load()

        # Determine start index
        start = self.collection_config.start_index

        # Determine end index
        total_instances = len(instances)
        if self.collection_config.max_instances:
            end = min(start + self.collection_config.max_instances, total_instances)
        else:
            end = total_instances

        stats.total = end - start
        logger.info(f"Collecting {stats.total} instances (starting from {start}) with {self.collection_config.max_workers} worker(s)")

        def process_instance(idx: int, instance: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
            instance_id = instance.get("instance_id", f"instance-{idx}")

            # Skip if already completed
            if instance_id in completed_ids:
                logger.info(f"Skipping already completed: {instance_id}")
                return instance_id, True, None

            # Check if trajectory file already exists
            trajectory_path = self.output_dir / f"{instance_id}.json"
            if trajectory_path.exists():
                logger.info(f"Trajectory already exists: {instance_id}")
                return instance_id, True, None

            # Collect trajectory
            trajectory_data = self.collect_instance(instance)
            return instance_id, False, trajectory_data

        collection_inputs = [
            (idx, instances[idx])
            for idx in range(start, end)
        ]

        completed_results = iter_parallel_completed(
            collection_inputs,
            lambda item: process_instance(item[0], item[1]),
            max_workers=self.collection_config.max_workers,
        )

        for instance_id, skipped, trajectory_data in completed_results:
            if skipped:
                stats.skipped += 1
                self.checkpoint.mark_completed(instance_id)
                continue

            if trajectory_data:
                # Save trajectory synchronously to avoid race conditions with I/O.
                self.save_trajectory(instance_id, trajectory_data)
                self.checkpoint.mark_completed(instance_id)
                stats.success += 1
                logger.info(f"Collected trajectory: {instance_id}")
                continue

            stats.failed += 1

        stats.end_time = datetime.now()

        # Log summary
        logger.info(f"Collection complete: {stats.to_dict()}")

        return stats.to_dict()


def print_rich_summary(collection_config: CollectionConfig, stats: dict[str, Any]) -> None:
    """Print collection summary with rich tables."""
    summary = Table(show_header=False, box=None)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Dataset", collection_config.dataset_name)
    summary.add_row("Output directory", collection_config.output_dir)
    summary.add_row("Total", str(stats["total"]))
    summary.add_row("Success", str(stats["success"]))
    summary.add_row("Failed", str(stats["failed"]))
    summary.add_row("Skipped", str(stats["skipped"]))
    if stats["duration_seconds"] is not None:
        summary.add_row("Duration", f"{stats['duration_seconds']:.1f}s")

    console.print()
    console.print(Panel(summary, title="Collection Summary", border_style="cyan"))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect agent trajectories from SWE-smith dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (threads).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in dataset",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=50,
        help="Maximum number of trajectories to collect.",
    )
    parser.add_argument(
        "--no-docker-filter",
        action="store_true",
        help="Disable filtering by available Docker images",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle the dataset before collection",
    )
    args = parser.parse_args()    # Load config
    config = Config.load(args.config)

    # Create collection config
    collection_config = CollectionConfig(
        dataset_name=config.data.train_dataset,
        output_dir=config.paths.trajectory_dir,
        max_instances=args.max_trajectories,
        max_workers=args.max_workers,
        start_index=args.start_index,
        split=config.data.train_split,
        filter_by_docker=not args.no_docker_filter,
        shuffle=args.shuffle,
        seed=config.sft.seed,
    )

    # Create collector and run
    collector = TrajectoryCollector(collection_config, config)
    stats = collector.collect_all()

    print_rich_summary(collection_config, stats)


if __name__ == "__main__":
    main()
