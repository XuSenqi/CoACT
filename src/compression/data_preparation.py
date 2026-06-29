"""Rollout data preparation pipeline for SFT training."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from litellm import acompletion

from src.agent.miniswe_runner import MiniSWERunner, create_runner_config_from_config
from src.agent.trajectory_parser import TrajectoryStep, parse_trajectory
from src.compression.anchor_reward import (
    AnchorAggregator,
    sample_anchor_set,
    score_candidate_action,
)
from src.compression.compression_filter import CompressionFilter, CompressionFilterReason
from src.compression.output_parser import interpret_compression_response
from src.compression.prompt_context import (
    build_step_compression_context,
    has_context_focus_question,
)
from src.config.config import Config
from src.config.prompts import render_compression_prompt
from src.reward.length_reward import compute_length_reward
from src.utils import CheckpointManager, gather_bounded

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Off-policy SFT data is written to its own file under paths.sft_data_dir so it
# can be merged with the DAGGER output via config.sft.data_file (a list).
OFFPOLICY_OUTPUT_FILENAME = "sft_data.offpolicy.jsonl"


@dataclass(frozen=True)
class RolloutSample:
    """One rollout sample before reward calculation."""

    compressed_text: str
    effective_text: str
    normalized_completion: str | None
    keep_original: bool
    valid_response: bool
    inferred_action: str
    step_id: int
    trajectory_id: str


class DataPreparationPipeline:
    """Pipeline for preparing raw rollout data for SFT training."""

    def __init__(
        self,
        config: Config,
        compression_max_workers: int = 8,
        inference_max_workers: int = 4,
        anchor_max_workers: int = 8,
        step_max_workers: int = 1,
    ) -> None:
        """Initialize the pipeline from config.

        Args:
            config: The unified configuration object.
            compression_max_workers: Maximum concurrent compression requests.
            inference_max_workers: Maximum concurrent action inference requests.
            anchor_max_workers: Maximum concurrent anchor-sampling agent calls.
            step_max_workers: Maximum concurrent trajectory steps.
        """
        if step_max_workers <= 0:
            raise ValueError(f"step_max_workers must be positive, got {step_max_workers}")

        self.config = config
        self.compression_max_workers = compression_max_workers
        self.inference_max_workers = inference_max_workers
        self.step_max_workers = step_max_workers

        rollout_cfg = config.sft.rollout
        data_prep_cfg = config.sft.data_preparation
        self.output_data_file = str(Path(config.paths.sft_data_dir) / OFFPOLICY_OUTPUT_FILENAME)

        self.num_samples = data_prep_cfg.num_samples
        self.min_similarity = data_prep_cfg.min_similarity

        self.compression_model = rollout_cfg.model
        self.compression_api_base = rollout_cfg.api_endpoint
        self.compression_api_key = rollout_cfg.api_key
        self.temperature = rollout_cfg.temperature
        self.top_p = rollout_cfg.top_p
        self.top_k = rollout_cfg.top_k
        self.min_p = rollout_cfg.min_p
        self.presence_penalty = rollout_cfg.presence_penalty
        self.repetition_penalty = rollout_cfg.repetition_penalty
        self.max_tokens = rollout_cfg.max_tokens
        self.request_timeout = rollout_cfg.request_timeout
        self.model_retry_stop_after_attempt = rollout_cfg.model_retry_stop_after_attempt

        self.compression_filter = CompressionFilter.build(data_prep_cfg.skip_compression_max_tokens)
        self.filter_counts: Counter[CompressionFilterReason] = Counter()

        # Agent runner (Qwen3.5-35B) - for reasoning with compressed context
        runner_config = create_runner_config_from_config(config)
        self.agent_runner = MiniSWERunner(runner_config)

        # Anchor runner: samples the agent's natural next action on the
        # uncompressed context at anchor_temperature (> 0) to build the anchor
        # set used to score action_reward (same reward the DAGGER prep uses).
        # continue_from_step never spins up a Docker env, so this runner stays
        # container-free.
        self.anchor_runner = MiniSWERunner(
            replace(runner_config, temperature=data_prep_cfg.anchor_temperature)
        )
        self.anchor_aggregator = AnchorAggregator.from_config(data_prep_cfg)
        self.anchor_num_samples = data_prep_cfg.anchor_num_samples
        self.anchor_max_workers = anchor_max_workers

        # Checkpoint manager
        checkpoint_path = Path(self.output_data_file).with_name(
            f"{Path(self.output_data_file).stem}.checkpoint.json"
        )
        self.checkpoint = CheckpointManager(checkpoint_path)
        self.raw_model_output_file = build_raw_model_output_path(self.output_data_file)

    async def generate_compressions(
        self,
        prompt: str,
        num_samples: int,
    ) -> list[str]:
        """Generate N compressed versions using Compressor model.

        Args:
            prompt: The compression prompt.
            num_samples: Number of samples to generate.

        Returns:
            List of compressed text outputs.
        """
        logger.info(
            f"Prompt for compression:\n{prompt[:500]}\n......\n{prompt[-500:]}\n"
            f"Generating {num_samples} compressed outputs..."
        )

        # Build kwargs for litellm.acompletion
        kwargs = {
            "model": self.compression_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "timeout": self.request_timeout,
            "num_retries": self.model_retry_stop_after_attempt,
        }

        # Add api_base if specified (for OpenAI-compatible endpoints)
        if self.compression_api_base:
            kwargs["api_base"] = self.compression_api_base
        if self.compression_api_key is not None:
            kwargs["api_key"] = self.compression_api_key

        # Add vLLM-specific extra_body for Qwen models
        if "qwen3.5" in self.compression_model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        async def _single_completion(i: int) -> str:
            try:
                response = await acompletion(**kwargs)
                content = response.choices[0].message.content or ""
                logger.info(f"Completion {i} finished. Content length: {len(content)}")
                if not content:
                    logger.warning(f"Completion {i} returned empty content! Response: {response}")
                return content
            except Exception as e:
                logger.warning(f"Failed to generate single compression: {e}")
                return ""

        results = await gather_bounded(
            list(range(num_samples)),
            _single_completion,
            self.compression_max_workers,
        )

        return [res for res in results if res]

    async def infer_action_with_compression(
        self,
        trajectory_messages: list[dict],
        step_index: int,
        compressed_output: str,
        returncode: int = 0,
        exception_info: str | None = None,
        use_original_output: bool = False,
    ) -> str | None:
        """Let Agent reason with compressed tool output.

        Uses MiniSWERunner.continue_from_step to inject the compressed
        output into the trajectory and have the agent continue reasoning.

        Note: This method is async to avoid blocking the event loop,
        but the underlying MiniSWERunner is synchronous (uses asyncio.to_thread).

        Args:
            trajectory_messages: The original trajectory's message list.
            step_index: The step index to replace tool output.
            compressed_output: Compressed tool output.
            returncode: The return code for the compressed output (default: 0).
            exception_info: Exception information if any (default: None).
            use_original_output: Whether to continue with the untouched original
                tool message instead of replacing it with compressed text.

        Returns:
            The bash command the agent would execute, or None if failed.
        """
        return await asyncio.to_thread(
            self.agent_runner.continue_from_step,
            trajectory_messages=trajectory_messages,
            step_index=step_index,
            compressed_output=compressed_output,
            returncode=returncode,
            exception_info=exception_info,
            use_original_output=use_original_output,
        )

    async def _infer_sample_action(
        self,
        index: int,
        compressed_text: str,
        step: TrajectoryStep,
        trajectory_messages: list[dict],
        trajectory_id: str,
    ) -> RolloutSample | None:
        """Infer the next action for one compressed output sample.

        Args:
            index: Sample index.
            compressed_text: The raw compression model output.
            step: Original trajectory step.
            trajectory_messages: Context messages.
            trajectory_id: ID for tracking.

        Returns:
            RolloutSample if successful, else `None`.
        """
        logger.info(f"Evaluating sample {index} with length {len(compressed_text)}")
        if not compressed_text.strip():
            logger.warning(f"Sample {index} is empty, skipping.")
            return None

        try:
            context = build_step_compression_context(step)
            resolution = interpret_compression_response(
                compressed_text,
                original_output=context["tool_output_for_prompt"],
                numbered_output=context["tool_output_for_prompt"],
            )

            if len(step.tool_output) != 1:
                logger.warning(
                    f"Step {step.message_index} has {len(step.tool_output)} tool outputs; "
                    "single-output compression expected"
                )
                return None

            tool_output = step.tool_output[0]
            returncode = tool_output.returncode
            exception_info = tool_output.exception

            # Use MiniSWERunner to continue from this step with compressed output
            inferred_action = await self.infer_action_with_compression(
                trajectory_messages=trajectory_messages,
                step_index=step.message_index,
                compressed_output=resolution.effective_output,
                returncode=returncode,
                exception_info=exception_info,
                use_original_output=resolution.keep_original,
            )

            if inferred_action is None:
                logger.warning(f"Could not infer action for sample {index}")
                return None

            return RolloutSample(
                compressed_text=compressed_text,
                effective_text=resolution.effective_output,
                normalized_completion=resolution.normalized_completion,
                keep_original=resolution.keep_original,
                valid_response=resolution.valid_response,
                inferred_action=inferred_action,
                step_id=step.message_index,
                trajectory_id=trajectory_id,
            )
        except Exception as e:
            logger.warning(f"Failed to infer action for sample {index}: {e}")
            return None

    async def process_step(
        self,
        step: TrajectoryStep,
        step_index: int,
        trajectory_messages: list[dict],
        trajectory_id: str,
        gt_tool_call: tuple[str, ...] | None = None,
    ) -> list[dict]:
        """Process a single trajectory step and return raw rollout examples.

        Args:
            step: TrajectoryStep object with goal, tool_output, tool_call, and
                context_focus_question.
            step_index: Index of the step in the trajectory.
            trajectory_messages: The original trajectory's message list.
            trajectory_id: Unique identifier for the trajectory.
            gt_tool_call: The next step's tool_call (used for scoring).

        Returns:
            List of raw rollout examples for later training-time selection.
        """
        # Skip steps without tool_output (nothing to compress)
        if not step.tool_output:
            logger.debug(f"Skipping step {step_index}: no tool_output")
            return []

        if len(step.tool_output) != 1 or len(step.tool_call) != 1:
            logger.info(
                f"Skipping step {step_index}: expected exactly one tool call and output, "
                f"got {len(step.tool_call)} calls and {len(step.tool_output)} outputs"
            )
            return []

        # Skip steps without context_focus_question
        if not has_context_focus_question(step.context_focus_question):
            logger.debug(f"Skipping step {step_index}: no context_focus_question")
            return []

        # Skip if no ground truth (last step)
        if not gt_tool_call:
            logger.debug(f"Skipping step {step_index}: no ground_truth (last step)")
            return []

        context = build_step_compression_context(step)
        tool_output_str = context["tool_output_for_prompt"]
        filter_reason = self.compression_filter.get_filter_reason(
            tool_calls=context["tool_call"],
            tool_output_str=tool_output_str,
            step_index=step.message_index,
        )
        if filter_reason is not None:
            self.filter_counts[filter_reason] += 1
            logger.info(
                f"Skipping step {step_index} for {trajectory_id} due to {filter_reason.value}",
            )
            return []

        prompt = render_compression_prompt(
            goal=context["goal"],
            context_focus_question=context["context_focus_question"],
            tool_output=tool_output_str,
            tool_call=context["tool_call"],
        )

        try:
            # Step 1: Generate N compressed versions
            compressed_outputs = await self.generate_compressions(
                prompt=prompt,
                num_samples=self.num_samples,
            )
            logger.info(f"Generated {len(compressed_outputs)} compressions.")
            if compressed_outputs:
                save_raw_model_outputs(
                    [
                        {
                            "trajectory_id": trajectory_id,
                            "step_id": step.message_index,
                            "sample_index": sample_index,
                            "prompt": prompt,
                            "raw_completion": raw_completion,
                        }
                        for sample_index, raw_completion in enumerate(compressed_outputs)
                    ],
                    self.raw_model_output_file,
                )
        except Exception as e:
            logger.error(f"Failed to generate compressions for step {step_index}: {e}")
            return []

        async def _evaluate_sample(item: tuple[int, str]) -> RolloutSample | None:
            index, compressed_text = item
            return await self._infer_sample_action(
                index=index,
                compressed_text=compressed_text,
                step=step,
                trajectory_messages=trajectory_messages,
                trajectory_id=trajectory_id,
            )

        results = await gather_bounded(
            list(enumerate(compressed_outputs)),
            _evaluate_sample,
            self.inference_max_workers,
        )
        rollout_samples = [r for r in results if r is not None]

        if not rollout_samples:
            logger.warning(f"All evaluated samples returned None for step {step_index}")
            return []

        retained_rollout_samples = [sample for sample in rollout_samples if sample.valid_response]
        dropped_fallback_count = len(rollout_samples) - len(retained_rollout_samples)
        if dropped_fallback_count > 0:
            logger.info(
                f"Dropping {dropped_fallback_count} invalid fallback samples from SFT data "
                f"for step {step_index}"
            )

        if not retained_rollout_samples:
            logger.warning(
                f"All rollout samples resolved to invalid fallback for step {step_index}; "
                "nothing will be written to SFT data"
            )
            return []

        # Score each candidate by how closely the agent's inferred next action
        # matches the agent-natural anchor set sampled on the uncompressed
        # context (the same reward definition the DAGGER prep uses).
        anchor_set = await sample_anchor_set(
            self.anchor_runner,
            prefix_messages=trajectory_messages,
            step_index=step.message_index,
            num_samples=self.anchor_num_samples,
            max_workers=self.anchor_max_workers,
        )
        if not anchor_set:
            logger.warning(
                f"Anchor set empty for {trajectory_id}#{step.message_index}; dropping step"
            )
            return []

        action_rewards = [
            score_candidate_action(sample.inferred_action, anchor_set, self.anchor_aggregator)
            for sample in retained_rollout_samples
        ]
        # Within-group length reward: among behavior-preserving candidates
        # (action_reward >= min_similarity), the shortest effective output gets
        # +0.5 and the longest -0.5. Selection keeps the top-length_reward k.
        length_rewards = compute_length_reward(
            compressed_outputs=[sample.effective_text for sample in retained_rollout_samples],
            action_rewards=action_rewards,
            theta=self.min_similarity,
        )

        examples: list[dict] = []
        for sample, action_reward, length_reward in zip(
            retained_rollout_samples, action_rewards, length_rewards, strict=True
        ):
            if sample.normalized_completion is None:
                raise ValueError("Valid rollout samples must have a normalized completion")
            examples.append(
                {
                    "prompt": prompt,
                    "completion": sample.normalized_completion,
                    "action_reward": action_reward,
                    "length_reward": length_reward,
                    "inferred_action": sample.inferred_action,
                    "ground_truth": list(gt_tool_call),
                    "anchor_size": len(anchor_set),
                    "anchor_aggregator": self.anchor_aggregator.kind,
                    # Namespace the group key so off-policy and DAGGER rows for
                    # the same instance/step never collide during selection.
                    "trajectory_id": f"offpolicy:{sample.trajectory_id}",
                    "step_id": sample.step_id,
                }
            )

        logger.info(f"Prepared {len(examples)} raw rollout samples for step {step_index}")

        return examples

    async def process_trajectory(
        self,
        trajectory_path: Path,
    ) -> list[dict]:
        """Process a single trajectory file.

        Args:
            trajectory_path: Path to trajectory JSON file.

        Returns:
            List of raw rollout examples.
        """
        logger.info(f"Processing trajectory: {trajectory_path}")

        with open(trajectory_path) as f:
            trajectory_json = json.load(f)

        trajectory = parse_trajectory(trajectory_json)
        trajectory_id = trajectory.instance_id
        trajectory_messages = trajectory_json.get("messages", [])

        if not trajectory_messages:
            logger.warning(f"No messages found in {trajectory_path}")
            return []

        all_examples = []
        assistant_steps = [
            (i, step) for i, step in enumerate(trajectory.steps) if step.role == "assistant"
        ]

        step_inputs: list[tuple[int, TrajectoryStep, tuple[str, ...] | None]] = []
        for idx, (step_index, step) in enumerate(assistant_steps):
            # Get next assistant step's tool_call as ground_truth
            ground_truth = None
            if idx + 1 < len(assistant_steps):
                next_step = assistant_steps[idx + 1][1]
                ground_truth = next_step.tool_call

            step_inputs.append((step_index, step, ground_truth))

        async def _process_step_input(
            item: tuple[int, TrajectoryStep, tuple[str, ...] | None],
        ) -> list[dict]:
            step_index, step, ground_truth = item
            return await self.process_step(
                step,
                step_index,
                trajectory_messages,
                trajectory_id,
                ground_truth,
            )

        step_results = await gather_bounded(
            step_inputs,
            _process_step_input,
            self.step_max_workers,
        )
        for examples in step_results:
            all_examples.extend(examples)

        logger.info(f"Generated {len(all_examples)} raw rollout examples from {trajectory_path}")
        return all_examples

    async def prepare_data(self) -> int:
        """Prepare raw rollout data from trajectory files.

        Returns:
            Total number of raw rollout examples generated.
        """
        trajectory_dir = Path(self.config.paths.trajectory_dir)
        if not trajectory_dir.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

        # Find all trajectory files
        trajectory_files = sorted(
            f for f in trajectory_dir.glob("*.json") if f.name != ".checkpoint.json"
        )
        if not trajectory_files:
            raise FileNotFoundError(f"No trajectory files found in {trajectory_dir}")

        logger.info(f"Found {len(trajectory_files)} trajectory files")

        # Load checkpoint
        completed_ids = self.checkpoint.load()

        # Process all trajectories and save immediately
        total_examples = 0
        for trajectory_path in trajectory_files:
            trajectory_id = trajectory_path.stem

            # Skip if already completed
            if trajectory_id in completed_ids:
                logger.info(f"Skipping already processed: {trajectory_id}")
                continue

            try:
                examples = await self.process_trajectory(trajectory_path)
                if examples:
                    save_dataset(examples, self.output_data_file)

                total_examples += len(examples)
                self.checkpoint.mark_completed(trajectory_id)
                logger.info(
                    f"Processed {trajectory_id}: {len(examples)} raw rollout examples "
                    f"(total: {total_examples})"
                )

            except Exception as e:
                logger.error(f"Failed to process {trajectory_path}: {e}")
                continue

        if total_examples == 0:
            raise ValueError("No raw rollout examples generated")

        if self.filter_counts:
            summary_parts = [
                f"{reason.value}={count}"
                for reason, count in sorted(
                    self.filter_counts.items(),
                    key=lambda item: item[0].value,
                )
            ]
            logger.info(f"Skipped steps summary: {', '.join(summary_parts)}")

        logger.info(f"Total raw rollout examples: {total_examples}")

        return total_examples


def save_dataset(examples: list[dict], output_path: str) -> None:
    """Save raw rollout examples to a JSON lines file.

    Args:
        examples: List of raw rollout examples.
        output_path: Output file path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def build_raw_model_output_path(output_path: str) -> Path:
    """Build the sibling JSONL path used to store raw compressor outputs.

    Args:
        output_path: Main raw-rollout dataset path.

    Returns:
        Path for the raw-model-output sidecar file.
    """
    dataset_path = Path(output_path)
    return dataset_path.with_name(f"{dataset_path.stem}.raw_model_outputs.jsonl")


def save_raw_model_outputs(records: list[dict], output_path: str | Path) -> None:
    """Append raw compressor outputs to a JSONL file.

    Args:
        records: Raw compressor output records to persist.
        output_path: Output file path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
