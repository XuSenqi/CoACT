"""Configuration loading and validation for CoACT.

Provides a Config class that loads configuration from YAML files and
validates all required fields and constraints.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Configuration error raised when validation fails."""

    pass


@dataclass(frozen=True)
class ModelPricingEntry:
    """Per-token pricing for a single model (USD)."""

    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    cache_read_input_token_cost: float = 0.0
    cache_creation_input_token_cost: float = 0.0


@dataclass(frozen=True)
class PricingConfig:
    """Pricing configuration mapping model names to their per-token costs."""

    models: dict[str, ModelPricingEntry] = field(default_factory=dict)

    def get_pricing(self, model: str) -> ModelPricingEntry:
        """Return pricing for a model, falling back to zero-cost defaults.

        Args:
            model: LiteLLM model identifier.

        Returns:
            ModelPricingEntry for the model, or all-zero defaults if not configured.
        """
        return self.models.get(model, ModelPricingEntry())


@dataclass(frozen=True)
class AgentConfig:
    """Agent model configuration."""

    model: str
    api_endpoint: str | None = None
    api_key: str | None = None

    # Inference parameters for tool calling
    temperature: float = 0.6
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 81920
    step_limit: int = 0
    request_timeout: float = 300.0
    model_retry_stop_after_attempt: int = 3

    def __post_init__(self) -> None:
        """Validate agent configuration."""
        if not self.model or not isinstance(self.model, str):
            raise ConfigError("agent.model must be a non-empty string")

        if self.api_endpoint is not None and not isinstance(self.api_endpoint, str):
            raise ConfigError("agent.api_endpoint must be a string or None")

        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ConfigError("agent.api_key must be a non-empty string or None")

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise ConfigError(f"agent.temperature must be non-negative, got {self.temperature}")

        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float)) or not (0 <= self.top_p <= 1)
        ):
            raise ConfigError(f"agent.top_p must be between 0 and 1, got {self.top_p}")

        if self.top_k is not None and (
            not isinstance(self.top_k, int) or (self.top_k != -1 and self.top_k <= 0)
        ):
            raise ConfigError(f"agent.top_k must be -1 (disabled) or positive, got {self.top_k}")

        if self.min_p is not None and (
            not isinstance(self.min_p, float) or not (0 <= self.min_p <= 1)
        ):
            raise ConfigError(f"agent.min_p must be between 0 and 1, got {self.min_p}")

        if self.presence_penalty is not None and (
            not isinstance(self.presence_penalty, float)
            or not (-2.0 <= self.presence_penalty <= 2.0)
        ):
            raise ConfigError(
                f"agent.presence_penalty must be between -2 and 2, got {self.presence_penalty}"
            )

        if self.repetition_penalty is not None and (
            not isinstance(self.repetition_penalty, float) or self.repetition_penalty <= 0
        ):
            raise ConfigError(
                f"agent.repetition_penalty must be positive, got {self.repetition_penalty}"
            )

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ConfigError(f"agent.max_tokens must be positive, got {self.max_tokens}")

        if not isinstance(self.step_limit, int) or self.step_limit < 0:
            raise ConfigError(f"agent.step_limit must be >= 0, got {self.step_limit}")

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(f"agent.request_timeout must be positive, got {self.request_timeout}")

        if (
            not isinstance(self.model_retry_stop_after_attempt, int)
            or self.model_retry_stop_after_attempt <= 0
        ):
            raise ConfigError(
                "agent.model_retry_stop_after_attempt must be positive, "
                f"got {self.model_retry_stop_after_attempt}"
            )


@dataclass(frozen=True)
class SFTRolloutConfig:
    """SFT rollout compression model configuration for LiteLLM.

    Supports any LiteLLM-compatible provider:
    - OpenAI: model="openai/gpt-4o-mini", api_endpoint not needed
    - Anthropic: model="anthropic/claude-3-5-haiku-latest", api_endpoint not needed
    - vLLM/Together/Fireworks/etc: model="openai/<model>", api_endpoint="<base_url>"
    - Local model wrappers exposed through LiteLLM-compatible endpoints
    """

    model: str
    """LiteLLM model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-5-haiku-latest')."""

    api_endpoint: str | None = None
    """API base URL for OpenAI-compatible endpoints (e.g., vLLM, Together, Fireworks)."""

    api_key: str | None = None
    """API key passed only to the SFT rollout model."""

    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 81920
    request_timeout: float = 120.0
    model_retry_stop_after_attempt: int = 3
    """Maximum tokens in response."""

    def __post_init__(self) -> None:
        """Validate SFT rollout compression configuration."""
        if not self.model or not isinstance(self.model, str):
            raise ConfigError("sft.rollout.model must be a non-empty string")

        if self.api_endpoint is not None and not isinstance(self.api_endpoint, str):
            raise ConfigError("sft.rollout.api_endpoint must be a string or None")

        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ConfigError("sft.rollout.api_key must be a non-empty string or None")

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise ConfigError(
                f"sft.rollout.temperature must be non-negative, got {self.temperature}"
            )

        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float)) or not (0 <= self.top_p <= 1)
        ):
            raise ConfigError(f"sft.rollout.top_p must be between 0 and 1, got {self.top_p}")

        if self.top_k is not None and (
            not isinstance(self.top_k, int) or (self.top_k != -1 and self.top_k <= 0)
        ):
            raise ConfigError(
                f"sft.rollout.top_k must be -1 (disabled) or positive, got {self.top_k}"
            )

        if self.min_p is not None and (
            not isinstance(self.min_p, (int, float)) or not (0 <= self.min_p <= 1)
        ):
            raise ConfigError(f"sft.rollout.min_p must be between 0 and 1, got {self.min_p}")

        if self.presence_penalty is not None and (
            not isinstance(self.presence_penalty, (int, float))
            or not (-2.0 <= self.presence_penalty <= 2.0)
        ):
            raise ConfigError(
                "sft.rollout.presence_penalty must be between -2 and 2, "
                f"got {self.presence_penalty}"
            )

        if self.repetition_penalty is not None and (
            not isinstance(self.repetition_penalty, (int, float)) or self.repetition_penalty <= 0
        ):
            raise ConfigError(
                f"sft.rollout.repetition_penalty must be positive, got {self.repetition_penalty}"
            )

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ConfigError(f"sft.rollout.max_tokens must be positive, got {self.max_tokens}")

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(
                f"sft.rollout.request_timeout must be positive, got {self.request_timeout}"
            )

        if (
            not isinstance(self.model_retry_stop_after_attempt, int)
            or self.model_retry_stop_after_attempt <= 0
        ):
            raise ConfigError(
                "sft.rollout.model_retry_stop_after_attempt must be positive, "
                f"got {self.model_retry_stop_after_attempt}"
            )


@dataclass(frozen=True)
class SFTConfig:
    """SFT training configuration for Phase 4."""

    # Training settings
    seed: int = 42
    learning_rate: float = 2.0e-4
    per_device_train_batch_size: int = 4
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 2
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_length: int = 32768

    # Evaluation and saving
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 1

    # Precision
    bf16: bool = True

    # Data path(s). A list of JSONL basenames under paths.sft_data_dir that are
    # concatenated at training time (e.g. off-policy + DAGGER).
    data_file: list[str] = field(default_factory=lambda: ["sft_data.jsonl"])
    drop_overlength_examples: bool = False
    dataloader_drop_last: bool = True

    # Evaluation split ratio
    eval_split_ratio: float = 0.1
    model_path: str = "model/Qwen3.5-4B"
    rollout: SFTRolloutConfig = field(
        default_factory=lambda: SFTRolloutConfig(
            model="openai/gemini-3-flash-preview-thinking",
            api_endpoint="https://ssvip.dmxapi.com/v1",
        )
    )
    data_preparation: "DataPreparationConfig" = field(
        default_factory=lambda: DataPreparationConfig()
    )
    lora: "LoRAConfig" = field(default_factory=lambda: LoRAConfig())

    def __post_init__(self) -> None:
        """Validate SFT configuration."""
        if not isinstance(self.seed, int):
            raise ConfigError(f"sft.seed must be an integer, got {self.seed}")

        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise ConfigError(f"sft.learning_rate must be positive, got {self.learning_rate}")

        if (
            not isinstance(self.per_device_train_batch_size, int)
            or self.per_device_train_batch_size <= 0
        ):
            raise ConfigError("sft.per_device_train_batch_size must be positive")

        if not isinstance(self.gradient_checkpointing, bool):
            raise ConfigError("sft.gradient_checkpointing must be a boolean")

        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps <= 0
        ):
            raise ConfigError("sft.gradient_accumulation_steps must be positive")

        if not isinstance(self.num_train_epochs, int) or self.num_train_epochs <= 0:
            raise ConfigError("sft.num_train_epochs must be positive")

        if not isinstance(self.lr_scheduler_type, str) or not self.lr_scheduler_type:
            raise ConfigError("sft.lr_scheduler_type must be a non-empty string")

        if not isinstance(self.warmup_ratio, float) or not (0 <= self.warmup_ratio <= 1):
            raise ConfigError(f"sft.warmup_ratio must be between 0 and 1, got {self.warmup_ratio}")

        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ConfigError(f"sft.weight_decay must be non-negative, got {self.weight_decay}")

        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ConfigError(f"sft.max_length must be positive, got {self.max_length}")

        if not isinstance(self.eval_steps, int) or self.eval_steps <= 0:
            raise ConfigError("sft.eval_steps must be positive")

        if not isinstance(self.save_steps, int) or self.save_steps <= 0:
            raise ConfigError("sft.save_steps must be positive")

        if not isinstance(self.logging_steps, int) or self.logging_steps <= 0:
            raise ConfigError("sft.logging_steps must be positive")

        if not isinstance(self.bf16, bool):
            raise ConfigError("sft.bf16 must be a boolean")

        if not isinstance(self.drop_overlength_examples, bool):
            raise ConfigError("sft.drop_overlength_examples must be a boolean")

        if not isinstance(self.dataloader_drop_last, bool):
            raise ConfigError("sft.dataloader_drop_last must be a boolean")

        if not isinstance(self.eval_split_ratio, float) or not (0 <= self.eval_split_ratio < 1):
            raise ConfigError(
                f"sft.eval_split_ratio must be between 0 and 1, got {self.eval_split_ratio}"
            )

        if not self.model_path or not isinstance(self.model_path, str):
            raise ConfigError("sft.model_path must be a non-empty string")

        if not isinstance(self.data_file, list) or not self.data_file:
            raise ConfigError("sft.data_file must be a non-empty list of filenames")
        if any(not isinstance(name, str) or not name for name in self.data_file):
            raise ConfigError("sft.data_file entries must be non-empty strings")


@dataclass(frozen=True)
class DataPreparationConfig:
    """Data preparation configuration for Phase 4 SFT data generation."""

    num_samples: int = 8

    # Selection settings
    select_top_k: int = 4
    selection_mode: str = "reward"
    """SFT rollout selection mode: ``reward``, ``length_only``, or
    ``action_only``. RQ3 reward ablations use ``length_only`` for w/o AP and
    ``action_only`` for w/o LR."""

    min_similarity: float = 0.65
    skip_compression_max_tokens: int = 256
    inject_unchanged_fallback: bool = False
    """When True, groups with no rollout passing ``min_similarity`` contribute a
    single synthetic ``{"type":"unchanged","content":null}`` completion so the
    compressor learns a conservative fallback on hard steps."""

    anchor_num_samples: int = 8
    """Number of natural-action samples drawn from the agent on the uncompressed
    context, used as the anchor set for relabeling ``action_reward``."""

    anchor_temperature: float = 0.7
    """Sampling temperature for the agent during anchor-set construction. Must
    be > 0 to obtain a distribution rather than the deterministic mode."""

    dagger_inferred_temperature: float = 0.0
    """Agent temperature when recomputing ``inferred_action`` on a DAGGER deploy
    prefix. Default 0 matches the original ``inferred_action`` semantics
    (agent's deterministic next move)."""

    anchor_aggregator: str = "top3"
    """How per-anchor similarities collapse into ``action_reward``: one of
    ``mean``, ``max``, ``top3``, ``count``."""

    anchor_count_threshold: float = 0.5
    """Similarity threshold used by the ``count`` aggregator."""

    def __post_init__(self) -> None:
        """Validate data preparation configuration."""
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ConfigError(
                f"sft.data_preparation.num_samples must be positive, got {self.num_samples}"
            )

        if not isinstance(self.select_top_k, int) or self.select_top_k <= 0:
            raise ConfigError(
                f"sft.data_preparation.select_top_k must be positive, got {self.select_top_k}"
            )

        if self.selection_mode not in {"reward", "length_only", "action_only"}:
            raise ConfigError(
                "sft.data_preparation.selection_mode must be one of "
                f"reward/length_only/action_only, got {self.selection_mode!r}"
            )

        if not isinstance(self.anchor_num_samples, int) or self.anchor_num_samples <= 1:
            raise ConfigError(
                "sft.data_preparation.anchor_num_samples must be > 1 to form a "
                f"distribution, got {self.anchor_num_samples}"
            )

        if not isinstance(self.anchor_temperature, float) or self.anchor_temperature <= 0:
            raise ConfigError(
                "sft.data_preparation.anchor_temperature must be > 0, got "
                f"{self.anchor_temperature}"
            )

        if (
            not isinstance(self.dagger_inferred_temperature, float)
            or self.dagger_inferred_temperature < 0
        ):
            raise ConfigError(
                "sft.data_preparation.dagger_inferred_temperature must be ≥ 0, got "
                f"{self.dagger_inferred_temperature}"
            )

        if self.anchor_aggregator not in {"mean", "max", "top3", "count"}:
            raise ConfigError(
                "sft.data_preparation.anchor_aggregator must be one of "
                f"mean/max/top3/count, got {self.anchor_aggregator!r}"
            )

        if not isinstance(self.anchor_count_threshold, (int, float)) or not (
            0 <= self.anchor_count_threshold <= 1
        ):
            raise ConfigError(
                "sft.data_preparation.anchor_count_threshold must be between 0 and 1, "
                f"got {self.anchor_count_threshold}"
            )

        if not isinstance(self.inject_unchanged_fallback, bool):
            raise ConfigError(
                "sft.data_preparation.inject_unchanged_fallback must be a boolean, "
                f"got {self.inject_unchanged_fallback!r}"
            )

        if not isinstance(self.min_similarity, float) or not (0 <= self.min_similarity <= 1):
            raise ConfigError(
                "sft.data_preparation.min_similarity must be between 0 and 1, "
                f"got {self.min_similarity}"
            )

        if (
            not isinstance(self.skip_compression_max_tokens, int)
            or self.skip_compression_max_tokens <= 0
        ):
            raise ConfigError(
                "sft.data_preparation.skip_compression_max_tokens must be positive, "
                f"got {self.skip_compression_max_tokens}"
            )


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA fine-tuning configuration."""

    r: int = 4
    alpha: int = 8
    dropout: float = 0.0
    target_modules: list[str] | str = "all-linear"
    use_rslora: bool = False
    use_dora: bool = False

    def __post_init__(self) -> None:
        """Validate LoRA configuration."""
        if not isinstance(self.r, int) or self.r <= 0:
            raise ConfigError(f"lora.r must be positive, got {self.r}")

        if not isinstance(self.alpha, int) or self.alpha <= 0:
            raise ConfigError(f"lora.alpha must be positive, got {self.alpha}")

        if not isinstance(self.dropout, (int, float)) or not (0 <= self.dropout <= 1):
            raise ConfigError(f"lora.dropout must be between 0 and 1, got {self.dropout}")

        if not isinstance(self.target_modules, (list, str)) or (
            isinstance(self.target_modules, list) and len(self.target_modules) == 0
        ):
            raise ConfigError("lora.target_modules must be a non-empty list or a string")


@dataclass(frozen=True)
class DataConfig:
    """Data configuration."""

    train_dataset: str = "data/SWE-smith"
    eval_dataset: str = "data/SWE-bench_Verified"
    train_split: str = "train"
    eval_split: str = "test"


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "CoACT"


@dataclass(frozen=True)
class PathsConfig:
    """Path configuration."""

    trajectory_dir: str = "data/trajectories"
    dagger_trajectory_dir: str = "data/trajectories_dagger"
    sft_data_dir: str = "data/sft"
    checkpoint_dir: str = "checkpoints"
    eval_run_dir: str = "data/eval/runs"


@dataclass(frozen=True)
class CoACTConfig:
    """CoACT trained compression model configuration."""

    model: str = "openai/CoACT-sft"
    api_endpoint: str = "http://localhost:8002/v1"
    temperature: float = 0.6
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 81920
    request_timeout: float = 120.0
    model_retry_stop_after_attempt: int = 3
    skip_compression_max_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ConfigError("evaluation.coact.model must be a non-empty string")

        if not self.api_endpoint or not isinstance(self.api_endpoint, str):
            raise ConfigError("evaluation.coact.api_endpoint must be a non-empty string")

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise ConfigError(
                f"evaluation.coact.temperature must be non-negative, got {self.temperature}"
            )

        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float)) or not (0 <= self.top_p <= 1)
        ):
            raise ConfigError(f"evaluation.coact.top_p must be between 0 and 1, got {self.top_p}")

        if self.top_k is not None and (
            not isinstance(self.top_k, int) or (self.top_k != -1 and self.top_k <= 0)
        ):
            raise ConfigError(
                f"evaluation.coact.top_k must be -1 (disabled) or positive, got {self.top_k}"
            )

        if self.min_p is not None and (
            not isinstance(self.min_p, (int, float)) or not (0 <= self.min_p <= 1)
        ):
            raise ConfigError(f"evaluation.coact.min_p must be between 0 and 1, got {self.min_p}")

        if self.presence_penalty is not None and (
            not isinstance(self.presence_penalty, (int, float))
            or not (-2.0 <= self.presence_penalty <= 2.0)
        ):
            raise ConfigError(
                "evaluation.coact.presence_penalty must be between -2 and 2, "
                f"got {self.presence_penalty}"
            )

        if self.repetition_penalty is not None and (
            not isinstance(self.repetition_penalty, (int, float)) or self.repetition_penalty <= 0
        ):
            raise ConfigError(
                "evaluation.coact.repetition_penalty must be positive, "
                f"got {self.repetition_penalty}"
            )

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ConfigError(f"evaluation.coact.max_tokens must be positive, got {self.max_tokens}")

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(
                f"evaluation.coact.request_timeout must be positive, got {self.request_timeout}"
            )

        if (
            not isinstance(self.model_retry_stop_after_attempt, int)
            or self.model_retry_stop_after_attempt <= 0
        ):
            raise ConfigError(
                "evaluation.coact.model_retry_stop_after_attempt must be positive, "
                f"got {self.model_retry_stop_after_attempt}"
            )

        if (
            not isinstance(self.skip_compression_max_tokens, int)
            or self.skip_compression_max_tokens <= 0
        ):
            raise ConfigError(
                "evaluation.coact.skip_compression_max_tokens must be positive, "
                f"got {self.skip_compression_max_tokens}"
            )


@dataclass(frozen=True)
class AgentDietConfig:
    """AgentDiet reflection model configuration for baseline evaluation."""

    model: str = "openai/gpt-5-mini"
    api_endpoint: str = "https://api.openai.com/v1"
    temperature: float = 0.6
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 81920
    request_timeout: float = 120.0
    model_retry_stop_after_attempt: int = 3
    delay_steps: int = 2
    window_before_steps: int = 1
    token_threshold: int = 500

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ConfigError("evaluation.agentdiet.model must be a non-empty string")

        if not self.api_endpoint or not isinstance(self.api_endpoint, str):
            raise ConfigError("evaluation.agentdiet.api_endpoint must be a non-empty string")

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise ConfigError(
                "evaluation.agentdiet.temperature must be non-negative, " f"got {self.temperature}"
            )

        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float)) or not (0 <= self.top_p <= 1)
        ):
            raise ConfigError(
                f"evaluation.agentdiet.top_p must be between 0 and 1, got {self.top_p}"
            )

        if self.top_k is not None and (
            not isinstance(self.top_k, int) or (self.top_k != -1 and self.top_k <= 0)
        ):
            raise ConfigError(
                "evaluation.agentdiet.top_k must be -1 (disabled) or positive, " f"got {self.top_k}"
            )

        if self.min_p is not None and (
            not isinstance(self.min_p, (int, float)) or not (0 <= self.min_p <= 1)
        ):
            raise ConfigError(
                f"evaluation.agentdiet.min_p must be between 0 and 1, got {self.min_p}"
            )

        if self.presence_penalty is not None and (
            not isinstance(self.presence_penalty, (int, float))
            or not (-2.0 <= self.presence_penalty <= 2.0)
        ):
            raise ConfigError(
                "evaluation.agentdiet.presence_penalty must be between -2 and 2, "
                f"got {self.presence_penalty}"
            )

        if self.repetition_penalty is not None and (
            not isinstance(self.repetition_penalty, (int, float)) or self.repetition_penalty <= 0
        ):
            raise ConfigError(
                "evaluation.agentdiet.repetition_penalty must be positive, "
                f"got {self.repetition_penalty}"
            )

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ConfigError(
                f"evaluation.agentdiet.max_tokens must be positive, got {self.max_tokens}"
            )

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(
                f"evaluation.agentdiet.request_timeout must be positive, got {self.request_timeout}"
            )

        if (
            not isinstance(self.model_retry_stop_after_attempt, int)
            or self.model_retry_stop_after_attempt <= 0
        ):
            raise ConfigError(
                "evaluation.agentdiet.model_retry_stop_after_attempt must be positive, "
                f"got {self.model_retry_stop_after_attempt}"
            )

        if not isinstance(self.delay_steps, int) or self.delay_steps < 0:
            raise ConfigError(
                f"evaluation.agentdiet.delay_steps must be >= 0, got {self.delay_steps}"
            )

        if not isinstance(self.window_before_steps, int) or self.window_before_steps < 0:
            raise ConfigError(
                "evaluation.agentdiet.window_before_steps must be >= 0, "
                f"got {self.window_before_steps}"
            )

        if not isinstance(self.token_threshold, int) or self.token_threshold < 0:
            raise ConfigError(
                "evaluation.agentdiet.token_threshold must be >= 0, " f"got {self.token_threshold}"
            )


@dataclass(frozen=True)
class SWEPrunerConfig:
    """SWEPruner pruning service configuration for baseline evaluation.

    SWEPruner runs as a separate FastAPI service hosting a fine-tuned
    Qwen3-Reranker (TokenScorer). The framework only talks to it over HTTP,
    so config knobs cover the endpoint, per-request pruning behavior, and
    the upstream ``min_chars`` short-output bypass threshold.
    """

    endpoint: str = "http://localhost:8003"
    threshold: float = 0.5
    always_keep_first_frags: bool = False
    chunk_overlap_tokens: int = 50
    request_timeout: float = 120.0
    retries: int = 3
    skip_compression_max_tokens: int = 500

    def __post_init__(self) -> None:
        if not self.endpoint or not isinstance(self.endpoint, str):
            raise ConfigError("evaluation.swepruner.endpoint must be a non-empty string")

        if not isinstance(self.threshold, (int, float)) or not (0.0 <= self.threshold <= 1.0):
            raise ConfigError(
                f"evaluation.swepruner.threshold must be between 0 and 1, got {self.threshold}"
            )

        if not isinstance(self.always_keep_first_frags, bool):
            raise ConfigError(
                "evaluation.swepruner.always_keep_first_frags must be a bool, "
                f"got {type(self.always_keep_first_frags).__name__}"
            )

        if not isinstance(self.chunk_overlap_tokens, int) or self.chunk_overlap_tokens < 0:
            raise ConfigError(
                "evaluation.swepruner.chunk_overlap_tokens must be >= 0, "
                f"got {self.chunk_overlap_tokens}"
            )

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(
                "evaluation.swepruner.request_timeout must be positive, "
                f"got {self.request_timeout}"
            )

        if not isinstance(self.retries, int) or self.retries <= 0:
            raise ConfigError(
                f"evaluation.swepruner.retries must be a positive int, got {self.retries}"
            )

        if (
            not isinstance(self.skip_compression_max_tokens, int)
            or self.skip_compression_max_tokens <= 0
        ):
            raise ConfigError(
                "evaluation.swepruner.skip_compression_max_tokens must be positive, "
                f"got {self.skip_compression_max_tokens}"
            )


@dataclass(frozen=True)
class LLMLingua2Config:
    """LLMLingua-2 task-agnostic prompt-compression service configuration.

    LLMLingua-2 is a token-classification compressor (XLM-RoBERTa-large,
    distilled from GPT-4). It is an encoder model and cannot be served by
    vLLM, so it runs as a separate FastAPI service launched via
    ``scripts/vllm/serve_llmlingua2.sh`` that wraps the official
    ``llmlingua.PromptCompressor``. The framework only talks to it over HTTP.

    Most defaults mirror the upstream ``compress_prompt_llmlingua2`` signature
    verbatim so the baseline stays faithful to the paper. The exceptions are
    ``rate`` (defaulted to 0.8 instead of the upstream 0.5: at 0.5 the dropped
    tokens turn code/tool output into unusable fragments, so we keep ~80% of
    tokens) and ``skip_compression_max_tokens`` (shared with the other
    baselines) which bypasses compression for short tool outputs.
    """

    endpoint: str = "http://localhost:8004"
    rate: float = 0.8
    target_token: int = -1
    force_tokens: tuple[str, ...] = ()
    force_reserve_digit: bool = False
    drop_consecutive: bool = False
    chunk_end_tokens: tuple[str, ...] = (".", "\n")
    use_token_level_filter: bool = True
    use_context_level_filter: bool = False
    request_timeout: float = 120.0
    retries: int = 3
    skip_compression_max_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.endpoint or not isinstance(self.endpoint, str):
            raise ConfigError("evaluation.llmlingua2.endpoint must be a non-empty string")

        if not isinstance(self.rate, (int, float)) or not (0.0 < self.rate <= 1.0):
            raise ConfigError(f"evaluation.llmlingua2.rate must be in (0, 1], got {self.rate}")

        if not isinstance(self.target_token, int) or (
            self.target_token != -1 and self.target_token <= 0
        ):
            raise ConfigError(
                "evaluation.llmlingua2.target_token must be -1 (disabled) or positive, "
                f"got {self.target_token}"
            )

        if not isinstance(self.force_tokens, tuple) or not all(
            isinstance(tok, str) for tok in self.force_tokens
        ):
            raise ConfigError("evaluation.llmlingua2.force_tokens must be a list of strings")

        if not isinstance(self.force_reserve_digit, bool):
            raise ConfigError(
                "evaluation.llmlingua2.force_reserve_digit must be a bool, "
                f"got {type(self.force_reserve_digit).__name__}"
            )

        if not isinstance(self.drop_consecutive, bool):
            raise ConfigError(
                "evaluation.llmlingua2.drop_consecutive must be a bool, "
                f"got {type(self.drop_consecutive).__name__}"
            )

        if not isinstance(self.chunk_end_tokens, tuple) or not all(
            isinstance(tok, str) for tok in self.chunk_end_tokens
        ):
            raise ConfigError("evaluation.llmlingua2.chunk_end_tokens must be a list of strings")

        if not isinstance(self.use_token_level_filter, bool):
            raise ConfigError(
                "evaluation.llmlingua2.use_token_level_filter must be a bool, "
                f"got {type(self.use_token_level_filter).__name__}"
            )

        if not isinstance(self.use_context_level_filter, bool):
            raise ConfigError(
                "evaluation.llmlingua2.use_context_level_filter must be a bool, "
                f"got {type(self.use_context_level_filter).__name__}"
            )

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(
                "evaluation.llmlingua2.request_timeout must be positive, "
                f"got {self.request_timeout}"
            )

        if not isinstance(self.retries, int) or self.retries <= 0:
            raise ConfigError(
                f"evaluation.llmlingua2.retries must be a positive int, got {self.retries}"
            )

        if (
            not isinstance(self.skip_compression_max_tokens, int)
            or self.skip_compression_max_tokens <= 0
        ):
            raise ConfigError(
                "evaluation.llmlingua2.skip_compression_max_tokens must be positive, "
                f"got {self.skip_compression_max_tokens}"
            )


@dataclass(frozen=True)
class LongCodeZipConfig:
    """LongCodeZip two-stage code-compression service configuration.

    LongCodeZip (Shi et al., 2025; arXiv:2510.00446) is a training-free,
    query-aware code compressor. Stage 1 ranks function-level chunks by
    conditional perplexity (Approximated Mutual Information) relative to the
    query; Stage 2 detects perplexity-based blocks inside the kept functions
    and selects them with a 0/1 knapsack under an adaptive budget. It needs a
    causal code LM to compute perplexity, so (like LLMLingua-2) it cannot be
    served by vLLM and runs as a separate FastAPI service launched via
    ``scripts/vllm/serve_longcodezip.sh`` wrapping the official ``LongCodeZip``
    class. The framework only talks to it over HTTP; the compressor model is a
    serve-script concern (``--model-name``), mirroring the SWEPruner and
    LLMLingua-2 baselines.

    All algorithm knobs default to the upstream ``compress_code_file``
    signature so the baseline stays faithful to the paper: full two-stage
    compression (``rank_only=False`` + ``use_knapsack=True``) with ``rate``
    being the token *retention* fraction. ``skip_compression_max_tokens`` is
    the shared short-output bypass used by the other baselines.
    """

    endpoint: str = "http://localhost:8005"
    rate: float = 0.5
    language: str = "python"
    dynamic_compression_ratio: float = 0.2
    context_budget: str = "+100"
    rank_only: bool = False
    fine_ratio: float | None = None
    fine_grained_importance_method: str = "conditional_ppl"
    min_lines_for_fine_grained: int = 5
    importance_beta: float = 0.5
    use_knapsack: bool = True
    request_timeout: float = 120.0
    retries: int = 3
    skip_compression_max_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.endpoint or not isinstance(self.endpoint, str):
            raise ConfigError("evaluation.longcodezip.endpoint must be a non-empty string")

        if not isinstance(self.rate, (int, float)) or not (0.0 < self.rate <= 1.0):
            raise ConfigError(f"evaluation.longcodezip.rate must be in (0, 1], got {self.rate}")

        if not self.language or not isinstance(self.language, str):
            raise ConfigError("evaluation.longcodezip.language must be a non-empty string")

        if not isinstance(self.dynamic_compression_ratio, (int, float)) or not (
            0.0 <= self.dynamic_compression_ratio <= 1.0
        ):
            raise ConfigError(
                "evaluation.longcodezip.dynamic_compression_ratio must be in [0, 1], "
                f"got {self.dynamic_compression_ratio}"
            )

        if not self.context_budget or not isinstance(self.context_budget, str):
            raise ConfigError("evaluation.longcodezip.context_budget must be a non-empty string")

        if not isinstance(self.rank_only, bool):
            raise ConfigError(
                "evaluation.longcodezip.rank_only must be a bool, "
                f"got {type(self.rank_only).__name__}"
            )

        if self.fine_ratio is not None and (
            not isinstance(self.fine_ratio, (int, float)) or not (0.0 < self.fine_ratio <= 1.0)
        ):
            raise ConfigError(
                "evaluation.longcodezip.fine_ratio must be null or in (0, 1], "
                f"got {self.fine_ratio}"
            )

        if self.fine_grained_importance_method not in (
            "conditional_ppl",
            "contrastive_perplexity",
        ):
            raise ConfigError(
                "evaluation.longcodezip.fine_grained_importance_method must be "
                "'conditional_ppl' or 'contrastive_perplexity', "
                f"got {self.fine_grained_importance_method!r}"
            )

        if (
            not isinstance(self.min_lines_for_fine_grained, int)
            or self.min_lines_for_fine_grained < 1
        ):
            raise ConfigError(
                "evaluation.longcodezip.min_lines_for_fine_grained must be a positive int, "
                f"got {self.min_lines_for_fine_grained}"
            )

        if not isinstance(self.importance_beta, (int, float)) or self.importance_beta < 0.0:
            raise ConfigError(
                f"evaluation.longcodezip.importance_beta must be >= 0, got {self.importance_beta}"
            )

        if not isinstance(self.use_knapsack, bool):
            raise ConfigError(
                "evaluation.longcodezip.use_knapsack must be a bool, "
                f"got {type(self.use_knapsack).__name__}"
            )

        if not isinstance(self.request_timeout, (int, float)) or self.request_timeout <= 0:
            raise ConfigError(
                "evaluation.longcodezip.request_timeout must be positive, "
                f"got {self.request_timeout}"
            )

        if not isinstance(self.retries, int) or self.retries <= 0:
            raise ConfigError(
                f"evaluation.longcodezip.retries must be a positive int, got {self.retries}"
            )

        if (
            not isinstance(self.skip_compression_max_tokens, int)
            or self.skip_compression_max_tokens <= 0
        ):
            raise ConfigError(
                "evaluation.longcodezip.skip_compression_max_tokens must be positive, "
                f"got {self.skip_compression_max_tokens}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation configuration."""

    # Sliding window: keep only the most recent N tool outputs
    sliding_window_size: int = 10

    # CoACT trained compression model
    coact: CoACTConfig = field(default_factory=CoACTConfig)

    # AgentDiet reflection baseline
    agentdiet: AgentDietConfig = field(default_factory=AgentDietConfig)

    # SWEPruner pruning-service baseline
    swepruner: SWEPrunerConfig = field(default_factory=SWEPrunerConfig)

    # LLMLingua-2 token-classification compression-service baseline
    llmlingua2: LLMLingua2Config = field(default_factory=LLMLingua2Config)

    # LongCodeZip two-stage code-compression-service baseline
    longcodezip: LongCodeZipConfig = field(default_factory=LongCodeZipConfig)

    # SWE-bench evaluation
    swebench_max_workers: int = 4
    instance_timeout: int = 1800

    def __post_init__(self) -> None:
        if not isinstance(self.sliding_window_size, int) or self.sliding_window_size <= 0:
            raise ConfigError(
                f"evaluation.sliding_window_size must be positive, got {self.sliding_window_size}"
            )
        if not isinstance(self.swebench_max_workers, int) or self.swebench_max_workers <= 0:
            raise ConfigError(
                "evaluation.swebench_max_workers must be positive, "
                f"got {self.swebench_max_workers}"
            )
        if not isinstance(self.instance_timeout, int) or self.instance_timeout <= 0:
            raise ConfigError(
                f"evaluation.instance_timeout must be positive, got {self.instance_timeout}"
            )


@dataclass(frozen=True)
class Config:
    """Main configuration class for CoACT."""

    agent: AgentConfig
    sft: SFTConfig = field(default_factory=SFTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)

    @staticmethod
    def _validate_allowed_keys(
        data: dict[str, Any],
        allowed_keys: set[str],
        section_name: str,
    ) -> None:
        """Raise when a config section contains keys outside the current schema."""
        unknown_keys = set(data) - allowed_keys
        if unknown_keys:
            keys = ", ".join(sorted(unknown_keys))
            raise ConfigError(f"Unknown {section_name} config keys: {keys}")

    @classmethod
    def load(cls, config_path: str | os.PathLike[str]) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Config: Loaded and validated configuration

        Raises:
            ConfigError: If configuration file is invalid
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse configuration: {e}") from e

        if not isinstance(config_data, dict):
            raise ConfigError("Configuration must be a dictionary")

        cls._validate_allowed_keys(
            config_data,
            {
                "agent",
                "sft",
                "data",
                "wandb",
                "paths",
                "evaluation",
                "pricing",
            },
            "root",
        )

        # Extract and validate agent configuration
        if "agent" not in config_data:
            raise ConfigError("Missing required section: agent")
        agent_data = config_data["agent"]
        if not isinstance(agent_data, dict):
            raise ConfigError("agent section must be a dictionary")
        if "model" not in agent_data:
            raise ConfigError("Missing required field: agent.model")

        agent_config = AgentConfig(
            model=agent_data["model"],
            api_endpoint=agent_data.get("api_endpoint"),
            api_key=agent_data.get("api_key"),
            # Inference parameters
            temperature=agent_data.get("temperature", 0.6),
            top_p=agent_data.get("top_p"),
            top_k=agent_data.get("top_k"),
            min_p=agent_data.get("min_p"),
            presence_penalty=agent_data.get("presence_penalty"),
            repetition_penalty=agent_data.get("repetition_penalty"),
            max_tokens=agent_data.get("max_tokens", 81920),
            step_limit=agent_data.get("step_limit", 0),
            request_timeout=agent_data.get("request_timeout", 300.0),
            model_retry_stop_after_attempt=agent_data.get("model_retry_stop_after_attempt", 3),
        )

        # Extract SFT configuration (optional)
        sft_data = config_data.get("sft", {})
        if sft_data and not isinstance(sft_data, dict):
            raise ConfigError("sft section must be a dictionary")
        cls._validate_allowed_keys(
            sft_data,
            {
                "model_path",
                "rollout",
                "seed",
                "learning_rate",
                "per_device_train_batch_size",
                "gradient_checkpointing",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "lr_scheduler_type",
                "warmup_ratio",
                "weight_decay",
                "max_length",
                "eval_split_ratio",
                "eval_steps",
                "save_steps",
                "logging_steps",
                "bf16",
                "data_file",
                "drop_overlength_examples",
                "dataloader_drop_last",
                "data_preparation",
                "lora",
            },
            "sft",
        )

        # Extract paths configuration (optional) - needed for sft data file path
        paths_data = config_data.get("paths", {})
        if paths_data and not isinstance(paths_data, dict):
            raise ConfigError("paths section must be a dictionary")

        paths_config = PathsConfig(
            trajectory_dir=paths_data.get("trajectory_dir", "data/trajectories"),
            dagger_trajectory_dir=paths_data.get(
                "dagger_trajectory_dir", "data/trajectories_dagger"
            ),
            sft_data_dir=paths_data.get("sft_data_dir", "data/sft"),
            checkpoint_dir=paths_data.get("checkpoint_dir", "checkpoints"),
            eval_run_dir=paths_data.get("eval_run_dir", "data/eval/runs"),
        )

        # Build full sft data file path(s) dynamically. Accept a scalar (legacy)
        # or a list and normalize to a list of full paths under sft_data_dir.
        raw_data_file = sft_data.get("data_file", "sft_data.jsonl")
        data_file_names = [raw_data_file] if isinstance(raw_data_file, str) else list(raw_data_file)
        sft_data_paths = [str(Path(paths_config.sft_data_dir) / name) for name in data_file_names]

        sft_rollout_data = sft_data.get("rollout", {})
        if sft_rollout_data and not isinstance(sft_rollout_data, dict):
            raise ConfigError("sft.rollout section must be a dictionary")
        cls._validate_allowed_keys(
            sft_rollout_data,
            {
                "model",
                "api_endpoint",
                "api_key",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "max_tokens",
                "request_timeout",
                "model_retry_stop_after_attempt",
            },
            "sft.rollout",
        )

        sft_rollout_config = SFTRolloutConfig(
            model=sft_rollout_data.get(
                "model",
                "openai/gemini-3-flash-preview-thinking",
            ),
            api_endpoint=sft_rollout_data.get(
                "api_endpoint",
                "https://ssvip.dmxapi.com/v1",
            ),
            api_key=sft_rollout_data.get("api_key"),
            temperature=sft_rollout_data.get("temperature", 1.0),
            top_p=sft_rollout_data.get("top_p"),
            top_k=sft_rollout_data.get("top_k"),
            min_p=sft_rollout_data.get("min_p"),
            presence_penalty=sft_rollout_data.get("presence_penalty"),
            repetition_penalty=sft_rollout_data.get("repetition_penalty"),
            max_tokens=sft_rollout_data.get("max_tokens", 81920),
            request_timeout=sft_rollout_data.get("request_timeout", agent_config.request_timeout),
            model_retry_stop_after_attempt=sft_rollout_data.get(
                "model_retry_stop_after_attempt",
                agent_config.model_retry_stop_after_attempt,
            ),
        )

        sft_data_preparation_data = sft_data.get("data_preparation", {})
        if sft_data_preparation_data and not isinstance(sft_data_preparation_data, dict):
            raise ConfigError("sft.data_preparation section must be a dictionary")

        sft_data_preparation_config = DataPreparationConfig(
            num_samples=sft_data_preparation_data.get("num_samples", 8),
            select_top_k=sft_data_preparation_data.get("select_top_k", 4),
            selection_mode=sft_data_preparation_data.get("selection_mode", "reward"),
            min_similarity=sft_data_preparation_data.get("min_similarity", 0.65),
            skip_compression_max_tokens=sft_data_preparation_data.get(
                "skip_compression_max_tokens", 256
            ),
            inject_unchanged_fallback=sft_data_preparation_data.get(
                "inject_unchanged_fallback", False
            ),
            anchor_num_samples=sft_data_preparation_data.get("anchor_num_samples", 8),
            anchor_temperature=sft_data_preparation_data.get("anchor_temperature", 0.7),
            dagger_inferred_temperature=sft_data_preparation_data.get(
                "dagger_inferred_temperature", 0.0
            ),
            anchor_aggregator=sft_data_preparation_data.get("anchor_aggregator", "top3"),
            anchor_count_threshold=sft_data_preparation_data.get("anchor_count_threshold", 0.5),
        )

        sft_lora_data = sft_data.get("lora", {})
        if sft_lora_data and not isinstance(sft_lora_data, dict):
            raise ConfigError("sft.lora section must be a dictionary")

        sft_lora_config = LoRAConfig(
            r=sft_lora_data.get("r", 4),
            alpha=sft_lora_data.get("alpha", 8),
            dropout=sft_lora_data.get("dropout", 0.0),
            target_modules=sft_lora_data.get("target_modules", "all-linear"),
            use_rslora=sft_lora_data.get("use_rslora", False),
            use_dora=sft_lora_data.get("use_dora", False),
        )

        sft_config = SFTConfig(
            seed=sft_data.get("seed", 42),
            learning_rate=sft_data.get("learning_rate", 2.0e-4),
            per_device_train_batch_size=sft_data.get("per_device_train_batch_size", 4),
            gradient_checkpointing=sft_data.get("gradient_checkpointing", True),
            gradient_accumulation_steps=sft_data.get("gradient_accumulation_steps", 4),
            num_train_epochs=sft_data.get("num_train_epochs", 3),
            lr_scheduler_type=sft_data.get("lr_scheduler_type", "cosine"),
            warmup_ratio=sft_data.get("warmup_ratio", 0.03),
            weight_decay=sft_data.get("weight_decay", 0.01),
            max_length=sft_data.get("max_length", 8192),
            eval_steps=sft_data.get("eval_steps", 100),
            save_steps=sft_data.get("save_steps", 500),
            logging_steps=sft_data.get("logging_steps", 10),
            bf16=sft_data.get("bf16", True),
            data_file=sft_data_paths,
            drop_overlength_examples=sft_data.get("drop_overlength_examples", False),
            dataloader_drop_last=sft_data.get("dataloader_drop_last", True),
            eval_split_ratio=sft_data.get("eval_split_ratio", 0.1),
            model_path=sft_data.get("model_path", "model/Qwen3.5-4B"),
            rollout=sft_rollout_config,
            data_preparation=sft_data_preparation_config,
            lora=sft_lora_config,
        )

        # Extract data configuration (optional)
        data_data = config_data.get("data", {})
        if data_data and not isinstance(data_data, dict):
            raise ConfigError("data section must be a dictionary")

        data_config = DataConfig(
            train_dataset=data_data.get("train_dataset", "data/SWE-smith"),
            eval_dataset=data_data.get("eval_dataset", "data/SWE-bench_Verified"),
            train_split=data_data.get("train_split", "train"),
            eval_split=data_data.get("eval_split", "test"),
        )

        # Extract wandb configuration (optional)
        wandb_data = config_data.get("wandb", {})
        if wandb_data and not isinstance(wandb_data, dict):
            raise ConfigError("wandb section must be a dictionary")

        wandb_config = WandbConfig(
            enabled=wandb_data.get("enabled", True),
            project=wandb_data.get("project", "CoACT"),
        )

        # Extract evaluation configuration (optional)
        eval_data = config_data.get("evaluation", {})
        if eval_data and not isinstance(eval_data, dict):
            raise ConfigError("evaluation section must be a dictionary")

        # Extract CoACT config
        coact_data = eval_data.get("coact", {})
        coact_config = CoACTConfig(
            model=coact_data.get("model", "openai/Qwen3.5-4B"),
            api_endpoint=coact_data.get("api_endpoint", "http://localhost:8002/v1"),
            temperature=coact_data.get("temperature", agent_config.temperature),
            top_p=coact_data.get("top_p"),
            top_k=coact_data.get("top_k"),
            min_p=coact_data.get("min_p"),
            presence_penalty=coact_data.get("presence_penalty"),
            repetition_penalty=coact_data.get("repetition_penalty"),
            max_tokens=coact_data.get("max_tokens", agent_config.max_tokens),
            request_timeout=coact_data.get("request_timeout", agent_config.request_timeout),
            model_retry_stop_after_attempt=coact_data.get(
                "model_retry_stop_after_attempt",
                agent_config.model_retry_stop_after_attempt,
            ),
            skip_compression_max_tokens=coact_data.get("skip_compression_max_tokens", 512),
        )

        agentdiet_data = eval_data.get("agentdiet", {})
        agentdiet_config = AgentDietConfig(
            model=agentdiet_data.get("model", "openai/gpt-5-mini"),
            api_endpoint=agentdiet_data.get("api_endpoint", "https://api.openai.com/v1"),
            temperature=agentdiet_data.get("temperature", agent_config.temperature),
            top_p=agentdiet_data.get("top_p"),
            top_k=agentdiet_data.get("top_k"),
            min_p=agentdiet_data.get("min_p"),
            presence_penalty=agentdiet_data.get("presence_penalty"),
            repetition_penalty=agentdiet_data.get("repetition_penalty"),
            max_tokens=agentdiet_data.get("max_tokens", agent_config.max_tokens),
            request_timeout=agentdiet_data.get("request_timeout", agent_config.request_timeout),
            model_retry_stop_after_attempt=agentdiet_data.get(
                "model_retry_stop_after_attempt",
                agent_config.model_retry_stop_after_attempt,
            ),
            delay_steps=agentdiet_data.get("delay_steps", 2),
            window_before_steps=agentdiet_data.get("window_before_steps", 1),
            token_threshold=agentdiet_data.get("token_threshold", 500),
        )

        swepruner_data = eval_data.get("swepruner", {})
        swepruner_config = SWEPrunerConfig(
            endpoint=swepruner_data.get("endpoint", "http://localhost:8003"),
            threshold=swepruner_data.get("threshold", 0.5),
            always_keep_first_frags=swepruner_data.get("always_keep_first_frags", False),
            chunk_overlap_tokens=swepruner_data.get("chunk_overlap_tokens", 50),
            request_timeout=swepruner_data.get("request_timeout", 120.0),
            retries=swepruner_data.get("retries", 3),
            skip_compression_max_tokens=swepruner_data.get("skip_compression_max_tokens", 500),
        )

        llmlingua2_data = eval_data.get("llmlingua2", {})
        llmlingua2_config = LLMLingua2Config(
            endpoint=llmlingua2_data.get("endpoint", "http://localhost:8004"),
            rate=llmlingua2_data.get("rate", 0.8),
            target_token=llmlingua2_data.get("target_token", -1),
            force_tokens=tuple(llmlingua2_data.get("force_tokens", ())),
            force_reserve_digit=llmlingua2_data.get("force_reserve_digit", False),
            drop_consecutive=llmlingua2_data.get("drop_consecutive", False),
            chunk_end_tokens=tuple(llmlingua2_data.get("chunk_end_tokens", (".", "\n"))),
            use_token_level_filter=llmlingua2_data.get("use_token_level_filter", True),
            use_context_level_filter=llmlingua2_data.get("use_context_level_filter", False),
            request_timeout=llmlingua2_data.get("request_timeout", 120.0),
            retries=llmlingua2_data.get("retries", 3),
            skip_compression_max_tokens=llmlingua2_data.get("skip_compression_max_tokens", 512),
        )

        longcodezip_data = eval_data.get("longcodezip", {})
        longcodezip_config = LongCodeZipConfig(
            endpoint=longcodezip_data.get("endpoint", "http://localhost:8005"),
            rate=longcodezip_data.get("rate", 0.5),
            language=longcodezip_data.get("language", "python"),
            dynamic_compression_ratio=longcodezip_data.get("dynamic_compression_ratio", 0.2),
            context_budget=longcodezip_data.get("context_budget", "+100"),
            rank_only=longcodezip_data.get("rank_only", False),
            fine_ratio=longcodezip_data.get("fine_ratio"),
            fine_grained_importance_method=longcodezip_data.get(
                "fine_grained_importance_method", "conditional_ppl"
            ),
            min_lines_for_fine_grained=longcodezip_data.get("min_lines_for_fine_grained", 5),
            importance_beta=longcodezip_data.get("importance_beta", 0.5),
            use_knapsack=longcodezip_data.get("use_knapsack", True),
            request_timeout=longcodezip_data.get("request_timeout", 120.0),
            retries=longcodezip_data.get("retries", 3),
            skip_compression_max_tokens=longcodezip_data.get("skip_compression_max_tokens", 512),
        )

        eval_config = EvaluationConfig(
            sliding_window_size=eval_data.get("sliding_window_size", 10),
            coact=coact_config,
            agentdiet=agentdiet_config,
            swepruner=swepruner_config,
            llmlingua2=llmlingua2_config,
            longcodezip=longcodezip_config,
            swebench_max_workers=eval_data.get("swebench_max_workers", 4),
            instance_timeout=eval_data.get("instance_timeout", 1800),
        )

        # Extract pricing configuration (optional)
        pricing_data = config_data.get("pricing", {})
        if pricing_data and not isinstance(pricing_data, dict):
            raise ConfigError("pricing section must be a dictionary")

        pricing_models: dict[str, ModelPricingEntry] = {}
        for model_name, price_data in (pricing_data or {}).items():
            if not isinstance(price_data, dict):
                raise ConfigError(f"pricing.{model_name} must be a dictionary")
            pricing_models[model_name] = ModelPricingEntry(
                input_cost_per_token=price_data.get("input_cost_per_token", 0.0),
                output_cost_per_token=price_data.get("output_cost_per_token", 0.0),
                cache_read_input_token_cost=price_data.get("cache_read_input_token_cost", 0.0),
                cache_creation_input_token_cost=price_data.get(
                    "cache_creation_input_token_cost", 0.0
                ),
            )
        pricing_config = PricingConfig(models=pricing_models)

        return cls(
            agent=agent_config,
            sft=sft_config,
            data=data_config,
            wandb=wandb_config,
            paths=paths_config,
            evaluation=eval_config,
            pricing=pricing_config,
        )

    def save(self, save_path: str | os.PathLike[str]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            save_path: Path where to save the configuration
        """
        config_dict = {
            "agent": {
                "model": self.agent.model,
                "api_endpoint": self.agent.api_endpoint,
                "api_key": self.agent.api_key,
                # Inference parameters
                "temperature": self.agent.temperature,
                "top_p": self.agent.top_p,
                "top_k": self.agent.top_k,
                "min_p": self.agent.min_p,
                "presence_penalty": self.agent.presence_penalty,
                "repetition_penalty": self.agent.repetition_penalty,
                "max_tokens": self.agent.max_tokens,
                "step_limit": self.agent.step_limit,
                "request_timeout": self.agent.request_timeout,
                "model_retry_stop_after_attempt": (self.agent.model_retry_stop_after_attempt),
            },
            "sft": {
                "model_path": self.sft.model_path,
                "rollout": {
                    "model": self.sft.rollout.model,
                    "api_endpoint": self.sft.rollout.api_endpoint,
                    "api_key": self.sft.rollout.api_key,
                    "temperature": self.sft.rollout.temperature,
                    "top_p": self.sft.rollout.top_p,
                    "top_k": self.sft.rollout.top_k,
                    "min_p": self.sft.rollout.min_p,
                    "presence_penalty": self.sft.rollout.presence_penalty,
                    "repetition_penalty": self.sft.rollout.repetition_penalty,
                    "max_tokens": self.sft.rollout.max_tokens,
                    "request_timeout": self.sft.rollout.request_timeout,
                    "model_retry_stop_after_attempt": (
                        self.sft.rollout.model_retry_stop_after_attempt
                    ),
                },
                "seed": self.sft.seed,
                "learning_rate": self.sft.learning_rate,
                "per_device_train_batch_size": self.sft.per_device_train_batch_size,
                "gradient_checkpointing": self.sft.gradient_checkpointing,
                "gradient_accumulation_steps": self.sft.gradient_accumulation_steps,
                "num_train_epochs": self.sft.num_train_epochs,
                "lr_scheduler_type": self.sft.lr_scheduler_type,
                "warmup_ratio": self.sft.warmup_ratio,
                "weight_decay": self.sft.weight_decay,
                "max_length": self.sft.max_length,
                "eval_split_ratio": self.sft.eval_split_ratio,
                "eval_steps": self.sft.eval_steps,
                "save_steps": self.sft.save_steps,
                "logging_steps": self.sft.logging_steps,
                "bf16": self.sft.bf16,
                "data_file": [Path(p).name for p in self.sft.data_file],  # Save basenames only
                "drop_overlength_examples": self.sft.drop_overlength_examples,
                "dataloader_drop_last": self.sft.dataloader_drop_last,
                "data_preparation": {
                    "num_samples": self.sft.data_preparation.num_samples,
                    "select_top_k": self.sft.data_preparation.select_top_k,
                    "selection_mode": self.sft.data_preparation.selection_mode,
                    "min_similarity": self.sft.data_preparation.min_similarity,
                    "skip_compression_max_tokens": (
                        self.sft.data_preparation.skip_compression_max_tokens
                    ),
                },
                "lora": {
                    "r": self.sft.lora.r,
                    "alpha": self.sft.lora.alpha,
                    "dropout": self.sft.lora.dropout,
                    "target_modules": self.sft.lora.target_modules,
                    "use_rslora": self.sft.lora.use_rslora,
                    "use_dora": self.sft.lora.use_dora,
                },
            },
            "data": {
                "train_dataset": self.data.train_dataset,
                "eval_dataset": self.data.eval_dataset,
                "train_split": self.data.train_split,
                "eval_split": self.data.eval_split,
            },
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
            },
            "paths": {
                "trajectory_dir": self.paths.trajectory_dir,
                "sft_data_dir": self.paths.sft_data_dir,
                "checkpoint_dir": self.paths.checkpoint_dir,
                "eval_run_dir": self.paths.eval_run_dir,
            },
            "evaluation": {
                "sliding_window_size": self.evaluation.sliding_window_size,
                "coact": {
                    "model": self.evaluation.coact.model,
                    "api_endpoint": self.evaluation.coact.api_endpoint,
                    "temperature": self.evaluation.coact.temperature,
                    "top_p": self.evaluation.coact.top_p,
                    "top_k": self.evaluation.coact.top_k,
                    "min_p": self.evaluation.coact.min_p,
                    "presence_penalty": self.evaluation.coact.presence_penalty,
                    "repetition_penalty": self.evaluation.coact.repetition_penalty,
                    "max_tokens": self.evaluation.coact.max_tokens,
                    "request_timeout": self.evaluation.coact.request_timeout,
                    "model_retry_stop_after_attempt": (
                        self.evaluation.coact.model_retry_stop_after_attempt
                    ),
                    "skip_compression_max_tokens": (
                        self.evaluation.coact.skip_compression_max_tokens
                    ),
                },
                "agentdiet": {
                    "model": self.evaluation.agentdiet.model,
                    "api_endpoint": self.evaluation.agentdiet.api_endpoint,
                    "temperature": self.evaluation.agentdiet.temperature,
                    "top_p": self.evaluation.agentdiet.top_p,
                    "top_k": self.evaluation.agentdiet.top_k,
                    "min_p": self.evaluation.agentdiet.min_p,
                    "presence_penalty": self.evaluation.agentdiet.presence_penalty,
                    "repetition_penalty": self.evaluation.agentdiet.repetition_penalty,
                    "max_tokens": self.evaluation.agentdiet.max_tokens,
                    "request_timeout": self.evaluation.agentdiet.request_timeout,
                    "model_retry_stop_after_attempt": (
                        self.evaluation.agentdiet.model_retry_stop_after_attempt
                    ),
                    "delay_steps": self.evaluation.agentdiet.delay_steps,
                    "window_before_steps": self.evaluation.agentdiet.window_before_steps,
                    "token_threshold": self.evaluation.agentdiet.token_threshold,
                },
                "swepruner": {
                    "endpoint": self.evaluation.swepruner.endpoint,
                    "threshold": self.evaluation.swepruner.threshold,
                    "always_keep_first_frags": self.evaluation.swepruner.always_keep_first_frags,
                    "chunk_overlap_tokens": self.evaluation.swepruner.chunk_overlap_tokens,
                    "request_timeout": self.evaluation.swepruner.request_timeout,
                    "retries": self.evaluation.swepruner.retries,
                    "skip_compression_max_tokens": (
                        self.evaluation.swepruner.skip_compression_max_tokens
                    ),
                },
                "llmlingua2": {
                    "endpoint": self.evaluation.llmlingua2.endpoint,
                    "rate": self.evaluation.llmlingua2.rate,
                    "target_token": self.evaluation.llmlingua2.target_token,
                    "force_tokens": list(self.evaluation.llmlingua2.force_tokens),
                    "force_reserve_digit": self.evaluation.llmlingua2.force_reserve_digit,
                    "drop_consecutive": self.evaluation.llmlingua2.drop_consecutive,
                    "chunk_end_tokens": list(self.evaluation.llmlingua2.chunk_end_tokens),
                    "use_token_level_filter": self.evaluation.llmlingua2.use_token_level_filter,
                    "use_context_level_filter": (
                        self.evaluation.llmlingua2.use_context_level_filter
                    ),
                    "request_timeout": self.evaluation.llmlingua2.request_timeout,
                    "retries": self.evaluation.llmlingua2.retries,
                    "skip_compression_max_tokens": (
                        self.evaluation.llmlingua2.skip_compression_max_tokens
                    ),
                },
                "swebench_max_workers": self.evaluation.swebench_max_workers,
                "instance_timeout": self.evaluation.instance_timeout,
            },
            "pricing": {
                model_name: {
                    "input_cost_per_token": entry.input_cost_per_token,
                    "output_cost_per_token": entry.output_cost_per_token,
                    "cache_read_input_token_cost": entry.cache_read_input_token_cost,
                    "cache_creation_input_token_cost": entry.cache_creation_input_token_cost,
                }
                for model_name, entry in self.pricing.models.items()
            },
        }

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated path.

        Args:
            path: Dot-separated path (e.g., "agent.model")
            default: Default value if path not found

        Returns:
            Configuration value at the path

        Raises:
            ConfigError: If path not found and no default provided
        """
        parts = path.split(".")
        value = self

        try:
            for part in parts:
                # Check if value is Config or one of the config dataclasses
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    raise AttributeError(f"Attribute '{part}' not found")
        except AttributeError as e:
            if default is not None:
                return default
            raise ConfigError(f"Configuration path not found: {path}") from e

        return value
