#!/usr/bin/env python3
"""SFT training script for CoACT Phase 4.

This script fine-tunes the compression base model (Qwen3.5-4B) using TRL's SFTTrainer
with LoRA adapters. It supports multi-GPU training via accelerate launch.

Usage:
    # Single GPU
    python scripts/train_sft.py --config config/config.yaml

    # Multi-GPU (4x A100)
    accelerate launch --config_file config/accelerate_config.yaml scripts/train_sft.py --config config/config.yaml

Input data format (raw rollout JSON lines):
    {
        "prompt": "You are a context compression assistant...\\n## Compressed Output\\n",
        "completion": "{\"type\":\"plain\",\"content\":\"...\"}",
        "action_reward": 0.9,
        "length_reward": 0.1,
        "inferred_action": "grep SESSION_TIMEOUT auth/config.py",
        "ground_truth": ["grep SESSION_TIMEOUT auth/config.py"],
        "trajectory_id": "traj-001",
        "step_id": 7
    }

Training-time selection is applied from the raw file using `action_reward`.
"""

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.compression import (
    create_prompt_completion_examples,
    select_rollout_samples,
    split_rollout_records_by_group,
)
from src.config.config import Config
from src.utils.chat import render_chat_prompt
from src.utils.dataset_cache import compute_cache_key, file_fingerprint, load_or_build

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_run_name(config: Config) -> str:
    """Build a descriptive W&B run name from the active training config."""
    model_name = Path(config.sft.model_path).name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    parts = [
        "sft",
        model_name,
        f"lr{config.sft.learning_rate:.1e}",
        config.sft.lr_scheduler_type,
        f"ep{config.sft.num_train_epochs}",
        f"bs{config.sft.per_device_train_batch_size}",
        f"ga{config.sft.gradient_accumulation_steps}",
        (
            f"len{config.sft.max_length // 1024}k"
            if config.sft.max_length % 1024 == 0
            else f"len{config.sft.max_length}"
        ),
        f"r{config.sft.lora.r}",
        f"s{config.sft.seed}",
    ]
    if config.sft.data_preparation.selection_mode != "reward":
        parts.append(f"sel{config.sft.data_preparation.selection_mode}")
    parts.append(timestamp)
    return "-".join(parts)


def resolve_sft_data_files(config: Config) -> list[Path]:
    """Resolve SFT data files from config paths.

    Args:
        config: The unified configuration object.

    Returns:
        Existing SFT data file paths.

    Raises:
        FileNotFoundError: If any configured file cannot be found.
    """
    resolved_paths: list[Path] = []
    for data_file in config.sft.data_file:
        configured_path = Path(data_file)
        candidates = [configured_path]
        if not configured_path.is_absolute():
            candidates.append(Path(config.paths.sft_data_dir) / configured_path)

        existing_paths = [candidate for candidate in candidates if candidate.exists()]
        if not existing_paths:
            raise FileNotFoundError(
                f"Training data not found for {data_file!r}. Checked: "
                f"{[str(candidate) for candidate in candidates]}"
            )
        resolved_paths.append(existing_paths[0])

    return resolved_paths


def load_sft_dataset(
    data_paths: Sequence[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    select_top_k: int,
    min_similarity: float,
    inject_unchanged_fallback: bool,
    selection_mode: str = "reward",
    eval_split_ratio: float = 0.1,
    seed: int = 42,
    drop_overlength_examples: bool = True,
) -> tuple[Dataset, Dataset]:
    """Load SFT dataset in plain-text prompt/completion format.

    Reads raw rollout JSONL from one or more files (concatenated), splits by
    group, selects top-k per step, applies the chat template to prompts,
    appends EOS to completions, and optionally drops overlength examples. TRL
    handles tokenization and completion masking via the plain-text
    ``prompt``/``completion`` columns.

    Args:
        data_paths: Paths to the raw rollout JSON lines data files. All files
            are concatenated; each must contain at least one record.
        tokenizer: The tokenizer for chat template and overlength filtering.
        max_length: Maximum sequence length used for training.
        select_top_k: Number of top-ranked samples to keep per step.
        min_similarity: Minimum total reward threshold.
        inject_unchanged_fallback: When True, groups with no rollout meeting
            ``min_similarity`` contribute one synthetic ``unchanged`` completion.
        selection_mode: Reward-ablation rollout selection mode.
        eval_split_ratio: Ratio of data to use for evaluation (default: 0.1).
        seed: Random seed for dataset split (default: 42).
        drop_overlength_examples: Whether to discard examples exceeding
            ``max_length`` instead of letting the trainer truncate them.

    Returns:
        Tuple of ``(train_dataset, eval_dataset)`` in plain-text format.
    """
    logger.info(f"Loading raw rollout data from {list(data_paths)}")
    raw_records: list[dict] = []
    for path in data_paths:
        with open(path, encoding="utf-8") as f:
            file_records = [json.loads(line) for line in f if line.strip()]
        if not file_records:
            raise ValueError(f"No raw rollout records found in {path}")
        logger.info(f"  {path}: {len(file_records)} records")
        raw_records.extend(file_records)

    train_records, eval_records = split_rollout_records_by_group(
        raw_records,
        eval_split_ratio=eval_split_ratio,
        seed=seed,
    )
    if not train_records:
        raise ValueError("No training groups available after group-level split")

    selected_train = select_rollout_samples(
        train_records,
        top_k=select_top_k,
        min_similarity=min_similarity,
        inject_unchanged_fallback=inject_unchanged_fallback,
        selection_mode=selection_mode,
    )
    if not selected_train:
        raise ValueError(
            "No training examples selected from raw rollout records. "
            "Check the reward distribution and selection thresholds."
        )
    selected_eval = select_rollout_samples(
        eval_records,
        top_k=select_top_k,
        min_similarity=min_similarity,
        inject_unchanged_fallback=inject_unchanged_fallback,
        selection_mode=selection_mode,
    )

    logger.info(
        f"Selected {len(selected_train)} train / {len(selected_eval)} eval "
        f"with selection_mode={selection_mode} "
        f"from {len(raw_records)} raw rollout records"
    )

    def to_plain_text(records: list[dict]) -> Dataset:
        examples = create_prompt_completion_examples(records)
        return Dataset.from_list(
            [
                {
                    "prompt": render_chat_prompt(ex["prompt"], tokenizer),
                    "completion": ex["completion"] + tokenizer.eos_token,
                }
                for ex in examples
            ]
        )

    train_dataset = to_plain_text(selected_train)
    eval_dataset = to_plain_text(selected_eval)

    if drop_overlength_examples:

        def _is_within_limit(example: dict) -> bool:
            return (
                len(
                    tokenizer.encode(
                        example["prompt"] + example["completion"],
                        add_special_tokens=False,
                    )
                )
                <= max_length
            )

        train_before = len(train_dataset)
        train_dataset = train_dataset.filter(
            _is_within_limit,
            desc=f"Filtering train > {max_length} tokens",
        )
        eval_before = len(eval_dataset)
        if len(eval_dataset) > 0:
            eval_dataset = eval_dataset.filter(
                _is_within_limit,
                desc=f"Filtering eval > {max_length} tokens",
            )
        logger.info(
            f"Dropped {train_before - len(train_dataset)} overlength train "
            f"({(train_before - len(train_dataset)) / max(train_before, 1):.2%}) and "
            f"{eval_before - len(eval_dataset)} eval samples"
        )

    logger.info(f"Train set: {len(train_dataset)} examples, Eval set: {len(eval_dataset)} examples")
    return train_dataset, eval_dataset


def train(config: Config, rebuild_cache: bool = False) -> None:
    """Run SFT training.

    Args:
        config: The unified configuration object.
        rebuild_cache: When True, discard any persisted dataset cache and
            rebuild from scratch.
    """
    run_name = build_run_name(config)

    # Setup paths
    output_dir = Path(config.paths.checkpoint_dir) / "sft" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data file(s) exist
    data_files = resolve_sft_data_files(config)

    logger.info(f"Loading base model from {config.sft.model_path}")
    logger.info(f"Training data: {[str(p) for p in data_files]}")

    if config.wandb.enabled and config.wandb.project:
        os.environ["WANDB_PROJECT"] = config.wandb.project
        logger.info(f"Setting WANDB_PROJECT={config.wandb.project}")
        logger.info(f"W&B run name: {run_name}")

    logger.info(f"Checkpoint output dir: {output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.sft.model_path,
        dtype=torch.bfloat16 if config.sft.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Persistent cache keeps the input dataset fingerprint stable across
    # runs, which is required for TRL's internal map-cache to hit.
    # Order-independent multi-file fingerprint so the cache key is stable
    # regardless of how the files are listed in config.
    sorted_data_files = sorted(data_files, key=lambda p: str(p.resolve()))
    cache_key = compute_cache_key(
        {
            "kind": "sft_plain_text",
            "data_files": [str(p.resolve()) for p in sorted_data_files],
            "data_files_fp": [file_fingerprint(p) for p in sorted_data_files],
            "select_top_k": config.sft.data_preparation.select_top_k,
            "selection_mode": config.sft.data_preparation.selection_mode,
            "min_similarity": config.sft.data_preparation.min_similarity,
            "inject_unchanged_fallback": config.sft.data_preparation.inject_unchanged_fallback,
            "eval_split_ratio": config.sft.eval_split_ratio,
            "seed": config.sft.seed,
            "max_length": config.sft.max_length,
            "drop_overlength": config.sft.drop_overlength_examples,
            "model_path": config.sft.model_path,
        }
    )
    cache_dir = data_files[0].parent / ".sft_cache" / cache_key

    def build_sft_datasets() -> tuple[Dataset, Dataset | None, dict[str, object]]:
        train_ds, eval_ds = load_sft_dataset(
            [str(p) for p in data_files],
            tokenizer,
            config.sft.max_length,
            config.sft.data_preparation.select_top_k,
            config.sft.data_preparation.min_similarity,
            config.sft.data_preparation.inject_unchanged_fallback,
            config.sft.data_preparation.selection_mode,
            config.sft.eval_split_ratio,
            config.sft.seed,
            config.sft.drop_overlength_examples,
        )
        # Normalize empty eval split to None so load_or_build skips saving it.
        eval_out: Dataset | None = eval_ds if len(eval_ds) > 0 else None
        meta = {
            "num_train": len(train_ds),
            "num_eval": len(eval_ds),
        }
        return train_ds, eval_out, meta

    train_dataset, eval_dataset, _ = load_or_build(
        cache_dir, build_sft_datasets, rebuild=rebuild_cache
    )

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config.sft.lora.r,
        lora_alpha=config.sft.lora.alpha,
        lora_dropout=config.sft.lora.dropout,
        target_modules=config.sft.lora.target_modules,
        use_rslora=config.sft.lora.use_rslora,
        use_dora=config.sft.lora.use_dora,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # Setup SFT training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        seed=config.sft.seed,
        per_device_train_batch_size=config.sft.per_device_train_batch_size,
        per_device_eval_batch_size=config.sft.per_device_train_batch_size,
        gradient_checkpointing=config.sft.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        dataloader_drop_last=config.sft.dataloader_drop_last,
        max_length=config.sft.max_length,
        learning_rate=config.sft.learning_rate,
        num_train_epochs=config.sft.num_train_epochs,
        lr_scheduler_type=config.sft.lr_scheduler_type,
        warmup_ratio=config.sft.warmup_ratio,
        weight_decay=config.sft.weight_decay,
        bf16=config.sft.bf16,
        logging_steps=config.sft.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.sft.eval_steps if eval_dataset is not None else None,
        save_steps=config.sft.save_steps,
        save_total_limit=3,
        report_to="wandb" if (config.wandb.enabled and config.wandb.project) else "none",
        run_name=run_name,
        completion_only_loss=True,
        use_liger_kernel=True,
        ddp_find_unused_parameters=False,
        packing=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_output_dir = output_dir / "final"
    logger.info(f"Saving model to {final_output_dir}")
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    logger.info("Training complete!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SFT Training for CoACT")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Discard any persisted SFT dataset cache and rebuild from scratch.",
    )
    args = parser.parse_args()

    # Load config
    config = Config.load(args.config)

    # Run training
    train(config, rebuild_cache=args.rebuild_cache)


if __name__ == "__main__":
    main()
