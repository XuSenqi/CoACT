"""Compression strategy implementations for evaluation baselines.

Each strategy lives in its own submodule. Public symbols are re-exported
here so existing callers can keep using ``from src.eval.baselines import X``.
"""

from src.config.config import Config

from .agentdiet import AgentDietCompression, AgentDietReductionResult
from .agentdiet_coact import AgentDietCoACTCompression
from .base import CompressionAttempt, CompressionStrategy
from .coact import CoACTCompression
from .llmlingua2 import LLMLingua2Compression
from .longcodezip import LongCodeZipCompression
from .sliding_window import SlidingWindow
from .sliding_window_coact import SlidingWindowCoACTCompression
from .swepruner import SWEPrunerCompression
from .vanilla import VanillaCompression

__all__ = [
    "AgentDietCompression",
    "AgentDietCoACTCompression",
    "AgentDietReductionResult",
    "CompressionAttempt",
    "CompressionStrategy",
    "LLMLingua2Compression",
    "LongCodeZipCompression",
    "VanillaCompression",
    "CoACTCompression",
    "SlidingWindow",
    "SlidingWindowCoACTCompression",
    "SWEPrunerCompression",
    "get_strategy",
]


def get_strategy(name: str, config: Config) -> CompressionStrategy:
    """Get a compression strategy by name.

    Args:
        name: Strategy name ("vanilla", "sliding_window", "CoACT",
            "sliding_window_CoACT", "agentdiet", "agentdiet_CoACT",
            "swepruner", "llmlingua2", "longcodezip").
        config: Configuration object.

    Returns:
        CompressionStrategy instance.

    Raises:
        ValueError: If strategy name is unknown.
    """
    if name == "vanilla":
        return VanillaCompression()
    if name == "sliding_window":
        return SlidingWindow(window_size=config.evaluation.sliding_window_size)
    if name == "CoACT":
        return CoACTCompression.from_config(config)
    if name == "sliding_window_CoACT":
        return SlidingWindowCoACTCompression.from_config(config)
    if name == "agentdiet":
        return AgentDietCompression.from_config(config)
    if name == "agentdiet_CoACT":
        return AgentDietCoACTCompression.from_config(config)
    if name == "swepruner":
        return SWEPrunerCompression.from_config(config)
    if name == "llmlingua2":
        return LLMLingua2Compression.from_config(config)
    if name == "longcodezip":
        return LongCodeZipCompression.from_config(config)
    raise ValueError(f"Unknown compression strategy: {name}")
