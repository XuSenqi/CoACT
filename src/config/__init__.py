"""Configuration module for CoACT.

Provides centralized configuration management for all components:
- Agent model configuration
- Compressor model configuration
- SFT training parameters
- LoRA settings
- Data paths
"""

from src.config.config import Config, ConfigError
from src.config.prompts import (
    render_compression_prompt,
)

__all__ = [
    "Config",
    "ConfigError",
    "render_compression_prompt",
]
