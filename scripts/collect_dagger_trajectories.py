"""Roll out deploy-style trajectories on SWE-smith using agent + M_0 compressor.

For each SWE-smith instance whose RAW (uncompressed) trajectory already exists
under ``config.paths.trajectory_dir``, run the agent + CoACT compressor (M_0
served at ``config.evaluation.coact.api_endpoint``) and save the resulting
trajectory under ``config.paths.dagger_trajectory_dir``. The patched
``CompressingAgent`` stashes each tool's pre-compression output as
``extra.original_tool_output``, which is what the downstream
``prepare_dagger_data.py`` consumes.

Resume-safe: skips instances whose dagger trajectory file already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from src.config.config import Config
from src.eval.baselines.coact import CoACTCompression
from src.eval.runner import EvaluationRunner

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--trajectory-max-workers",
        type=int,
        default=4,
        help="Parallel agent rollouts (each spins up its own docker container).",
    )
    parser.add_argument(
        "--limit-instances",
        type=int,
        default=0,
        help="If > 0, only roll out the first N instances (for smoke tests).",
    )
    return parser.parse_args()


def load_smith_instances(
    dataset_path: str,
    split: str,
    wanted_ids: set[str],
) -> list[dict]:
    """Load SWE-smith instances filtered to wanted_ids.

    SWE-smith rows already carry ``image_name`` directly, unlike SWE-bench
    Verified which builds it via ``make_test_spec``.
    """
    dataset = load_dataset(dataset_path, split=split)
    instances: list[dict] = []
    for raw in dataset:
        instance = dict(raw)
        if instance.get("instance_id") in wanted_ids:
            instances.append(instance)
    return instances


def main() -> None:
    args = parse_args()
    config = Config.load(args.config)

    raw_traj_dir = Path(config.paths.trajectory_dir)
    out_dir = Path(config.paths.dagger_trajectory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_ids = {p.stem for p in raw_traj_dir.glob("*.json") if p.stem != ".checkpoint"}
    if not wanted_ids:
        raise FileNotFoundError(
            f"No raw trajectories found under {raw_traj_dir}; nothing to mirror"
        )
    logger.info(f"Found {len(wanted_ids)} raw trajectories under {raw_traj_dir}")

    cached_ids = {p.stem for p in out_dir.glob("*.json")}
    if cached_ids:
        logger.info(f"{len(cached_ids)} dagger trajectories already exist; will skip")
    pending_ids = wanted_ids - cached_ids
    if not pending_ids:
        logger.info("All trajectories already collected. Nothing to do.")
        return

    instances = load_smith_instances(
        config.data.train_dataset,
        split=config.data.train_split,
        wanted_ids=pending_ids,
    )
    found_ids = {inst["instance_id"] for inst in instances}
    missing = pending_ids - found_ids
    if missing:
        logger.warning(
            f"{len(missing)} pending instances were not found in the SWE-smith "
            f"dataset (e.g. {sorted(missing)[:3]}); they will be skipped"
        )

    if args.limit_instances > 0:
        instances = instances[: args.limit_instances]
        logger.info(f"Limiting to first {len(instances)} pending instances")

    if not instances:
        logger.info("Nothing to roll out after filtering.")
        return

    strategy = CoACTCompression.from_config(config)
    runner = EvaluationRunner(
        config=config,
        strategy=strategy,
        output_dir=str(out_dir),
    )

    trajectory_paths, metrics = runner.run_dataset(
        instances=instances,
        trajectory_max_workers=args.trajectory_max_workers,
    )

    logger.info(
        f"Done. Wrote {len(trajectory_paths)} trajectories to {out_dir}. "
        f"Submissions={metrics.successful_submissions}/{metrics.total_instances}"
    )

    # Sanity: ensure the patch took — check at least one trajectory has
    # original_tool_output stashed somewhere.
    if trajectory_paths:
        with open(trajectory_paths[0], encoding="utf-8") as f:
            traj = json.load(f)
        has_field = any(
            isinstance(m.get("extra"), dict) and "original_tool_output" in m["extra"]
            for m in traj.get("messages", [])
            if m.get("role") == "tool"
        )
        if not has_field:
            logger.warning(
                f"Sanity check failed: {trajectory_paths[0]} has no tool message "
                "with extra.original_tool_output. The CompressingAgent patch may "
                "not have been applied. Downstream DAGGER prep will fail."
            )
        else:
            logger.info("Sanity check OK: original_tool_output present on tool messages.")


if __name__ == "__main__":
    main()
