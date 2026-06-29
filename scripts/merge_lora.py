#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base causal LM and save the result.

Usage:
    python scripts/merge_lora.py --base-model model/Qwen3.5-4B --adapter checkpoints/sft/.../final --output checkpoints/merged/Qwen3.5-4B-sft-merged
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "auto": "auto",
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path or HF id of the base model.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save the merged model.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Torch dtype used when loading the base model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers.from_pretrained.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    dtype = DTYPE_MAP[dtype_name]
    return None if dtype == "auto" else dtype


def load_base_wrapper_config(base_model: str) -> dict:
    config_path = Path(base_model) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Base model config not found: {config_path}")
    return json.loads(config_path.read_text())


def resize_base_model_embeddings(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
) -> bool:
    """Resize base-model token embeddings to match the tokenizer vocabulary.

    Args:
        model: Base causal LM loaded from the base checkpoint.
        tokenizer: Tokenizer selected for the merged model.

    Returns:
        ``True`` if the model embeddings were resized, otherwise ``False``.

    Raises:
        TypeError: If the model does not expose token embeddings correctly.
        ValueError: If the model does not provide input embeddings.
    """
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise ValueError("Model returned no input embeddings, cannot align vocab size")

    target_vocab_size = len(tokenizer)
    current_vocab_size = input_embeddings.weight.shape[0]
    if current_vocab_size == target_vocab_size:
        return False

    logger.info(
        f"Resizing base model token embeddings from {current_vocab_size} "
        f"to {target_vocab_size} to match the adapter tokenizer"
    )
    model.resize_token_embeddings(target_vocab_size)
    return True


def build_merged_wrapper_config(base_config: dict, merged_text_config: dict) -> dict:
    wrapper_config = dict(base_config)
    wrapper_config["model_type"] = base_config.get("model_type", "qwen3_5")
    wrapper_config["architectures"] = base_config.get(
        "architectures", ["Qwen3_5ForConditionalGeneration"]
    )
    wrapper_config["text_config"] = merged_text_config
    if "tie_word_embeddings" in merged_text_config:
        wrapper_config["tie_word_embeddings"] = merged_text_config["tie_word_embeddings"]
    return wrapper_config


def main() -> None:
    from peft import PeftModel

    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_dtype(args.dtype)
    base_wrapper_config = load_base_wrapper_config(args.base_model)

    logger.info(f"Loading tokenizer from {args.adapter}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        trust_remote_code=args.trust_remote_code,
    )

    logger.info(f"Loading base model from {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    resize_base_model_embeddings(base_model, tokenizer)

    logger.info(f"Loading adapter from {args.adapter}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    logger.info("Merging LoRA weights into the base model")
    merged_model = peft_model.merge_and_unload()

    logger.info(f"Saving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    merged_wrapper_config = build_merged_wrapper_config(
        base_wrapper_config,
        merged_model.config.to_dict(),
    )
    (output_dir / "config.json").write_text(
        json.dumps(merged_wrapper_config, indent=2, ensure_ascii=True) + "\n"
    )
    logger.info("Rewrote merged config.json with Qwen3.5 wrapper config for vLLM compatibility")
    logger.info("Merge complete")


if __name__ == "__main__":
    main()
