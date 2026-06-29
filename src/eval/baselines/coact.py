"""CoACT SFT-trained compression strategy."""

from src.config.config import Config

from .model_compression import _ModelCompressionStrategy


class CoACTCompression(_ModelCompressionStrategy):
    """CoACT SFT-trained compression model (LoRA adapter on Qwen3.5-4B)."""

    name = "CoACT"

    @classmethod
    def from_config(cls, config: Config) -> "CoACTCompression":
        """Create from configuration.

        Args:
            config: The unified configuration.

        Returns:
            CoACTCompression instance.
        """
        return cls(
            model=config.evaluation.coact.model,
            api_endpoint=config.evaluation.coact.api_endpoint,
            temperature=config.evaluation.coact.temperature,
            top_p=config.evaluation.coact.top_p,
            top_k=config.evaluation.coact.top_k,
            min_p=config.evaluation.coact.min_p,
            presence_penalty=config.evaluation.coact.presence_penalty,
            repetition_penalty=config.evaluation.coact.repetition_penalty,
            max_tokens=config.evaluation.coact.max_tokens,
            timeout=config.evaluation.coact.request_timeout,
            max_retries=config.evaluation.coact.model_retry_stop_after_attempt,
        )
