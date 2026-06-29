"""Test that config/config.yaml is valid and complete."""

from pathlib import Path

from src.config.config import Config


def test_config_yaml_exists() -> None:
    """Test that config.yaml exists."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    assert config_path.exists(), f"config.yaml not found at {config_path}"


def test_config_yaml_loads() -> None:
    """Test that config.yaml loads without errors."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config.load(str(config_path))
    assert config is not None


def test_config_yaml_has_required_sections() -> None:
    """Test that config.yaml has all required sections."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config.load(str(config_path))

    # Check required sections
    assert config.agent is not None
    assert config.sft is not None
    assert config.sft.rollout is not None
    assert config.sft.model_path
    assert config.sft.lora is not None
    assert config.sft.data_preparation is not None
    assert config.data is not None
    assert config.wandb is not None
    assert config.paths is not None


def test_config_yaml_data_preparation_values() -> None:
    """Test that data preparation configuration has valid values."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = Config.load(str(config_path))

    # Check values are valid (type/range) rather than exact defaults
    assert config.sft.data_preparation.num_samples > 0
    assert config.sft.data_preparation.select_top_k > 0
    assert 0 <= config.sft.data_preparation.min_similarity <= 1.0
    assert config.sft.data_preparation.skip_compression_max_tokens > 0
