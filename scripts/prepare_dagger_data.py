"""Build SFT training data from on-policy DAGGER deploy trajectories.

For each ``(deploy_trajectory, compressed_step)`` pair found under
``config.paths.dagger_trajectory_dir``:

1. Recover the ORIGINAL (uncompressed) tool output from
   ``extra.original_tool_output`` (stashed by the patched ``CompressingAgent``).
2. Render the compressor prompt for that step's CFQ + original output.
3. Generate K=8 fresh Gemini compression candidates.
4. Build the deploy prefix (the trajectory messages up to and including this
   asst) with the immediate tool message's ``content`` rewritten back to the
   raw original output. This is the prefix every downstream agent call uses.
5. Anchor: K=8 agent samples on this prefix at temp ``anchor_temperature``.
6. Inferred: per candidate, agent's deterministic next action on the prefix
   with the candidate's ``effective_output`` injected (temp
   ``dagger_inferred_temperature``).
7. Reward: ``aggregate(sim(inferred, anchor) for anchor in anchor_set)``,
   default ``top3`` mean.
8. Filter ``action_reward >= sft.data_preparation.min_similarity`` and write
   surviving rows to ``data/sft/sft_data.dagger.jsonl``.

The schema mirrors ``sft_data.jsonl`` so the existing SFT trainer can consume
the file unchanged. Resume-safe at the (trajectory_id, step_id) granularity.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from src.agent.miniswe_runner import MiniSWERunner, create_runner_config_from_config
from src.agent.trajectory_parser import (
    ToolOutput,
    Trajectory,
    TrajectoryStep,
    parse_trajectory,
)
from src.compression.anchor_reward import (
    AnchorAggregator,
    sample_anchor_set,
    score_candidate_action,
)
from src.compression.compression_filter import CompressionFilter
from src.compression.data_preparation import DataPreparationPipeline, save_dataset
from src.compression.output_parser import interpret_compression_response
from src.compression.prompt_context import (
    build_step_compression_context,
    has_context_focus_question,
)
from src.config.config import Config
from src.config.prompts import render_compression_prompt
from src.reward.length_reward import compute_length_reward
from src.utils import gather_bounded

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (defaults to <sft_data_dir>/sft_data.dagger.jsonl).",
    )
    parser.add_argument(
        "--compression-max-workers",
        type=int,
        default=8,
        help="Parallel Gemini compression calls.",
    )
    parser.add_argument(
        "--inference-max-workers",
        type=int,
        default=8,
        help="Parallel agent calls (anchor + inferred).",
    )
    parser.add_argument(
        "--step-max-workers",
        type=int,
        default=1,
        help="Maximum number of compressed steps to process concurrently.",
    )
    parser.add_argument(
        "--limit-trajectories",
        type=int,
        default=0,
        help="If > 0, process only the first N trajectories (for smoke tests).",
    )
    return parser.parse_args()


def load_completed_steps(output_path: Path) -> set[tuple[str, int]]:
    """Return (trajectory_id, step_id) pairs already present in the output."""
    completed: set[tuple[str, int]] = set()
    if not output_path.exists():
        return completed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            completed.add((r["trajectory_id"], int(r["step_id"])))
    return completed


def find_compressed_steps(
    trajectory: Trajectory,
    messages: list[dict],
) -> list[tuple[TrajectoryStep, int, list[str]]]:
    """Find asst steps whose tool messages were compressed by M_0.

    Returns a list of ``(step, tool_msg_index, original_outputs)`` tuples. Only
    steps where every following tool message in the asst's batch carries
    ``extra.original_tool_output`` (i.e. went through CompressingAgent) and
    whose content actually differs from that original (i.e. compression
    materially changed the message) are returned.
    """
    out: list[tuple[TrajectoryStep, int, list[str]]] = []
    for step in trajectory.steps:
        if step.role != "assistant":
            continue
        if not step.tool_call:
            continue
        if len(step.tool_call) != 1 or len(step.tool_output) != 1:
            logger.info(
                f"Skipping DAGGER step {step.message_index}: expected exactly one "
                f"tool call and output, got {len(step.tool_call)} calls and "
                f"{len(step.tool_output)} outputs"
            )
            continue
        # Find the FIRST tool message after this asst (the patch stashes for
        # all batch tools but downstream we only need the first one's prefix).
        first_tool_idx: int | None = None
        for j in range(step.message_index + 1, len(messages)):
            if messages[j].get("role") == "tool":
                first_tool_idx = j
                break
        if first_tool_idx is None:
            continue
        tool_msg = messages[first_tool_idx]
        extra = tool_msg.get("extra") or {}
        original = extra.get("original_tool_output")
        if not isinstance(original, str) or not original:
            continue
        rendered_content = tool_msg.get("content")
        if isinstance(rendered_content, str) and original.strip() == rendered_content.strip():
            continue
        out.append((step, first_tool_idx, [original]))
    return out


def synthesise_step_with_original(step: TrajectoryStep, original_output: str) -> TrajectoryStep:
    """Return a TrajectoryStep whose tool_output reflects the uncompressed text.

    Used so ``build_step_compression_context`` produces a prompt against the
    pre-compression tool output instead of the deploy-rendered (compressed)
    string already in the trajectory.
    """
    synthetic_tool = ToolOutput(returncode=0, output=original_output, exception=None)
    return replace(step, tool_output=(synthetic_tool,))


def messages_with_original_at(
    messages: list[dict],
    tool_msg_index: int,
    original_output: str,
) -> list[dict]:
    """Deep-copy messages with the tool message at index swapped to the original."""
    cloned = copy.deepcopy(messages)
    target = cloned[tool_msg_index]
    target["content"] = original_output
    extra = dict(target.get("extra") or {})
    extra["raw_output"] = original_output
    target["extra"] = extra
    return cloned


class DaggerStepProcessor:
    """Async processor for one (deploy_trajectory, asst_step) → list[record]."""

    def __init__(
        self,
        config: Config,
        gemini_pipeline: DataPreparationPipeline,
        anchor_runner: MiniSWERunner,
        inferred_runner: MiniSWERunner,
        compression_filter: CompressionFilter,
        anchor_max_workers: int,
        inference_max_workers: int,
    ) -> None:
        self.config = config
        self.gemini_pipeline = gemini_pipeline
        self.anchor_runner = anchor_runner
        self.inferred_runner = inferred_runner
        self.compression_filter = compression_filter
        self.K = config.sft.data_preparation.anchor_num_samples
        self.num_samples = config.sft.data_preparation.num_samples
        self.min_similarity = config.sft.data_preparation.min_similarity
        self.anchor_max_workers = anchor_max_workers
        self.inference_max_workers = inference_max_workers
        self.anchor_aggregator = AnchorAggregator.from_config(config.sft.data_preparation)

    async def process_step(
        self,
        trajectory_id: str,
        deploy_messages: list[dict],
        asst_step: TrajectoryStep,
        first_tool_idx: int,
        original_tool_output: str,
    ) -> list[dict]:
        if not has_context_focus_question(asst_step.context_focus_question):
            return []
        synthetic_step = synthesise_step_with_original(asst_step, original_tool_output)
        context = build_step_compression_context(synthetic_step)
        tool_output_for_prompt = context["tool_output_for_prompt"]
        filter_reason = self.compression_filter.get_filter_reason(
            tool_calls=context["tool_call"],
            tool_output_str=tool_output_for_prompt,
            step_index=asst_step.message_index,
        )
        if filter_reason is not None:
            logger.debug(
                f"Skipping {trajectory_id}#{asst_step.message_index}: {filter_reason.value}"
            )
            return []

        prompt = render_compression_prompt(
            goal=context["goal"],
            context_focus_question=context["context_focus_question"],
            tool_output=tool_output_for_prompt,
            tool_call=context["tool_call"],
        )

        try:
            candidates = await self.gemini_pipeline.generate_compressions(
                prompt=prompt,
                num_samples=self.num_samples,
            )
        except Exception as e:
            logger.exception(
                f"Gemini compression failed for {trajectory_id}#{asst_step.message_index}: {e}"
            )
            return []
        candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
        if not candidates:
            logger.warning(
                f"No valid Gemini candidates for {trajectory_id}#{asst_step.message_index}"
            )
            return []

        # Build the prefix with the tool message swapped back to its original.
        anchor_prefix = messages_with_original_at(
            deploy_messages, first_tool_idx, original_tool_output
        )

        anchor_set = await sample_anchor_set(
            self.anchor_runner,
            prefix_messages=anchor_prefix,
            step_index=asst_step.message_index,
            num_samples=self.K,
            max_workers=self.anchor_max_workers,
        )
        if not anchor_set:
            logger.warning(
                f"Anchor set empty for {trajectory_id}#{asst_step.message_index}; dropping step"
            )
            return []

        records = await self._score_candidates(
            candidates=candidates,
            anchor_prefix=anchor_prefix,
            asst_index=asst_step.message_index,
            tool_output_for_prompt=tool_output_for_prompt,
            original_tool_output=original_tool_output,
            anchor_set=anchor_set,
            prompt=prompt,
            trajectory_id=trajectory_id,
        )
        return [r for r in records if r["action_reward"] >= self.min_similarity]

    async def _score_candidates(
        self,
        *,
        candidates: list[str],
        anchor_prefix: list[dict],
        asst_index: int,
        tool_output_for_prompt: str,
        original_tool_output: str,
        anchor_set: list[str],
        prompt: str,
        trajectory_id: str,
    ) -> list[dict]:
        async def evaluate_one(item: tuple[int, str]) -> tuple[dict, str] | None:
            sample_index, raw_completion = item
            resolution = interpret_compression_response(
                raw_completion,
                original_output=original_tool_output,
                numbered_output=tool_output_for_prompt,
            )
            if not resolution.valid_response:
                return None
            inferred = await asyncio.to_thread(
                self.inferred_runner.continue_from_step,
                trajectory_messages=anchor_prefix,
                step_index=asst_index,
                compressed_output=resolution.effective_output,
                returncode=0,
                exception_info=None,
                use_original_output=resolution.keep_original,
            )
            if not isinstance(inferred, str) or not inferred.strip():
                return None
            reward = score_candidate_action(inferred, anchor_set, self.anchor_aggregator)
            record = {
                "prompt": prompt,
                "completion": resolution.normalized_completion,
                "action_reward": float(reward),
                # Filled in across the group below, once every candidate is scored.
                "length_reward": 0.0,
                "inferred_action": inferred,
                "anchor_set": anchor_set,
                "anchor_aggregator": self.anchor_aggregator.kind,
                "anchor_size": len(anchor_set),
                "trajectory_id": trajectory_id,
                "step_id": asst_index,
                "sample_index": sample_index,
                # DAGGER scores against the anchor set, not a single GT command;
                # leave this empty for schema compatibility with sft_data.jsonl.
                "ground_truth": [],
            }
            return record, resolution.effective_output

        results = await gather_bounded(
            list(enumerate(candidates)),
            evaluate_one,
            self.inference_max_workers,
        )
        scored = [r for r in results if r is not None]
        if not scored:
            return []
        records = [record for record, _ in scored]
        # Within-group length reward: prefer the shortest behavior-preserving
        # compression (action_reward >= min_similarity).
        length_rewards = compute_length_reward(
            compressed_outputs=[effective_output for _, effective_output in scored],
            action_rewards=[record["action_reward"] for record in records],
            theta=self.min_similarity,
        )
        for record, length_reward in zip(records, length_rewards, strict=True):
            record["length_reward"] = length_reward
        return records


@dataclass(frozen=True)
class ProcessedDaggerStep:
    """Result for one processed DAGGER step."""

    step: TrajectoryStep
    records: list[dict]
    failed: bool = False


async def process_pending_steps(
    *,
    processor: DaggerStepProcessor,
    trajectory_id: str,
    deploy_messages: list[dict],
    pending: list[tuple[TrajectoryStep, int, list[str]]],
    step_max_workers: int,
) -> list[ProcessedDaggerStep]:
    """Process pending DAGGER steps concurrently while isolating failures.

    Args:
        processor: Step processor used to generate and score records.
        trajectory_id: Source-namespaced trajectory identifier.
        deploy_messages: Raw deploy trajectory messages.
        pending: Pending ``(step, first_tool_idx, originals)`` tuples.
        step_max_workers: Maximum number of concurrent step workers.

    Returns:
        Per-step results in the same order as ``pending``.
    """

    async def process_one(
        item: tuple[TrajectoryStep, int, list[str]],
    ) -> ProcessedDaggerStep:
        step, first_tool_idx, originals = item
        try:
            records = await processor.process_step(
                trajectory_id=trajectory_id,
                deploy_messages=deploy_messages,
                asst_step=step,
                first_tool_idx=first_tool_idx,
                original_tool_output=originals[0],
            )
            return ProcessedDaggerStep(step=step, records=records)
        except Exception as e:
            logger.exception(f"Step {trajectory_id}#{step.message_index} failed: {e}")
            return ProcessedDaggerStep(step=step, records=[], failed=True)

    return await gather_bounded(pending, process_one, step_max_workers)


async def main_async() -> None:
    args = parse_args()
    if args.step_max_workers <= 0:
        raise ValueError(f"step_max_workers must be positive, got {args.step_max_workers}")
    config = Config.load(args.config)
    logger.info(f"Step workers: {args.step_max_workers}")

    output_path = args.output or Path(config.paths.sft_data_dir) / "sft_data.dagger.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_dir = Path(config.paths.dagger_trajectory_dir)
    if not trajectory_dir.exists():
        raise FileNotFoundError(
            f"Dagger trajectory dir not found: {trajectory_dir}. "
            "Run scripts/collect_dagger_trajectories.py first."
        )
    traj_files = sorted(p for p in trajectory_dir.glob("*.json") if p.name != ".checkpoint.json")
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files in {trajectory_dir}")
    if args.limit_trajectories > 0:
        traj_files = traj_files[: args.limit_trajectories]

    completed = load_completed_steps(output_path)
    if completed:
        logger.info(f"Resume: {len(completed)} (traj, step) pairs already in {output_path}")

    gemini_pipeline = DataPreparationPipeline(
        config,
        compression_max_workers=args.compression_max_workers,
        inference_max_workers=args.inference_max_workers,
    )

    base_runner_config = create_runner_config_from_config(config)
    anchor_runner = MiniSWERunner(
        replace(base_runner_config, temperature=config.sft.data_preparation.anchor_temperature)
    )
    inferred_runner = MiniSWERunner(
        replace(
            base_runner_config,
            temperature=config.sft.data_preparation.dagger_inferred_temperature,
        )
    )
    compression_filter = CompressionFilter.build(
        config.sft.data_preparation.skip_compression_max_tokens
    )

    processor = DaggerStepProcessor(
        config=config,
        gemini_pipeline=gemini_pipeline,
        anchor_runner=anchor_runner,
        inferred_runner=inferred_runner,
        compression_filter=compression_filter,
        anchor_max_workers=args.inference_max_workers,
        inference_max_workers=args.inference_max_workers,
    )

    total_written = 0
    for traj_path in traj_files:
        with open(traj_path, encoding="utf-8") as f:
            traj_json = json.load(f)
        messages = traj_json.get("messages") or []
        if not messages:
            logger.warning(f"Skipping {traj_path}: no messages")
            continue
        trajectory = parse_trajectory(traj_json)
        # Namespace the group key so off-policy and DAGGER rows for the same
        # instance/step never collide during selection. This id flows into the
        # written records, the resume-completed set, and logs consistently.
        trajectory_id = f"dagger:{trajectory.instance_id}"
        compressed_steps = find_compressed_steps(trajectory, messages)
        pending = [
            (s, idx, origs)
            for (s, idx, origs) in compressed_steps
            if (trajectory_id, s.message_index) not in completed
        ]
        if not pending:
            logger.info(
                f"{trajectory_id}: all {len(compressed_steps)} compressed steps already done"
            )
            continue
        logger.info(
            f"{trajectory_id}: processing {len(pending)}/{len(compressed_steps)} pending steps"
        )

        processed_steps = await process_pending_steps(
            processor=processor,
            trajectory_id=trajectory_id,
            deploy_messages=messages,
            pending=pending,
            step_max_workers=args.step_max_workers,
        )

        for processed_step in processed_steps:
            step = processed_step.step
            if processed_step.failed:
                continue
            if processed_step.records:
                save_dataset(processed_step.records, str(output_path))
                total_written += len(processed_step.records)
                logger.info(
                    f"{trajectory_id}#{step.message_index}: wrote "
                    f"{len(processed_step.records)} records (running total: {total_written})"
                )
            completed.add((trajectory_id, step.message_index))

    logger.info(f"Done. Wrote {total_written} new records to {output_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
