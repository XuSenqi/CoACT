"""Test configuration module for CoACT.

Tests follow TDD methodology:
1. Write test first (RED)
2. Run test - verify it FAILS
3. Implement minimal code (GREEN)
4. Run test - verify it PASSES
5. Refactor (IMPROVE)
6. Verify coverage (80%+)
"""

import os
import tempfile

import pytest
import yaml

from src.config.config import Config, ConfigError


class TestConfigLoading:
    """Test configuration file loading and parsing."""

    def test_load_valid_config(self) -> None:
        """Test loading a valid configuration file."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
                "api_key": "agent-key",
            },
            "sft": {
                "model_path": "model/Qwen3.5-4B",
                "rollout": {
                    "model": "openai/Qwen3.5-4B",
                    "api_endpoint": "http://localhost:8001/v1",
                    "api_key": "rollout-key",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.agent.model == "openai/Qwen3.5-35B-A3B-FP8"
            assert config.agent.api_endpoint == "http://localhost:8000/v1"
            assert config.agent.api_key == "agent-key"
            assert config.sft.rollout.model == "openai/Qwen3.5-4B"
            assert config.sft.rollout.api_key == "rollout-key"
        finally:
            os.unlink(config_path)

    def test_load_empty_api_keys_fail_fast(self) -> None:
        """Empty API keys should fail during configuration validation."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
                "api_key": "",
            },
            "sft": {
                "rollout": {
                    "api_key": "",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="agent.api_key must be a non-empty string"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_load_sft_dataloader_drop_last(self) -> None:
        """Test SFT dataloader_drop_last is parsed from YAML."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {
                "dataloader_drop_last": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.sft.dataloader_drop_last is False
        finally:
            os.unlink(config_path)

    def test_load_valid_provider_agent_config(self) -> None:
        """Test loading an agent config that uses a direct LiteLLM model."""
        config_data = {
            "agent": {
                "model": "openai/gpt-4o-mini",
                "api_endpoint": None,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.agent.model == "openai/gpt-4o-mini"
            assert config.agent.api_endpoint is None
        finally:
            os.unlink(config_path)

    def test_load_preserves_dmxapi_root_endpoint(self) -> None:
        """Test loading keeps a configured DMXAPI root URL unchanged."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "evaluation": {
                "agentdiet": {
                    "model": "openai/gpt-5-mini",
                    "api_endpoint": "https://www.dmxapi.cn",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.evaluation.agentdiet.api_endpoint == "https://www.dmxapi.cn"
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent configuration file raises error."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            Config.load("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="Failed to parse configuration"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_load_empty_config(self) -> None:
        """Test loading empty configuration raises validation error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="Missing required section"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_load_config_missing_required_fields(self) -> None:
        """Test loading config with missing required fields raises error."""
        config_data = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="Missing required section: agent"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_data_file_accepts_list(self) -> None:
        """sft.data_file accepts a YAML list and resolves under sft_data_dir."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {"data_file": ["a.jsonl", "b.jsonl"]},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.sft.data_file == ["data/sft/a.jsonl", "data/sft/b.jsonl"]
        finally:
            os.unlink(config_path)

    def test_data_file_scalar_normalizes_to_list(self) -> None:
        """A legacy scalar data_file normalizes to a one-element list."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {"data_file": "only.jsonl"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.sft.data_file == ["data/sft/only.jsonl"]
        finally:
            os.unlink(config_path)

    def test_anchor_aggregator_parsed_and_validated(self) -> None:
        """anchor_aggregator parses; an unknown kind fails fast."""
        good = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {
                "data_preparation": {
                    "anchor_aggregator": "mean",
                    "anchor_count_threshold": 0.3,
                }
            },
        }
        bad = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {"data_preparation": {"anchor_aggregator": "bogus"}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(good, f)
            good_path = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad, f)
            bad_path = f.name

        try:
            config = Config.load(good_path)
            assert config.sft.data_preparation.anchor_aggregator == "mean"
            assert config.sft.data_preparation.anchor_count_threshold == 0.3
            with pytest.raises(ConfigError, match="anchor_aggregator"):
                Config.load(bad_path)
        finally:
            os.unlink(good_path)
            os.unlink(bad_path)


class TestNestedConfigAccess:
    """Test accessing nested configuration values."""

    def test_access_deeply_nested_agent_config(self) -> None:
        """Test accessing deeply nested agent configuration."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
                "api_key": "agent-key",
            },
            "sft": {
                "rollout": {
                    "api_key": "rollout-key",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.agent.model == "openai/Qwen3.5-35B-A3B-FP8"
        finally:
            os.unlink(config_path)

    def test_access_nested_sft_config(self) -> None:
        """Test accessing SFT training configuration."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {
                "seed": 42,
                "learning_rate": 2.0e-4,
                "per_device_train_batch_size": 4,
                "max_length": 8192,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.sft.seed == 42
            assert config.sft.learning_rate == 2.0e-4
            assert config.sft.per_device_train_batch_size == 4
            assert config.sft.max_length == 8192
        finally:
            os.unlink(config_path)

    def test_access_nested_lora_config(self) -> None:
        """Test accessing LoRA configuration."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {
                "lora": {
                    "r": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": "all-linear",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.sft.lora.r == 4
            assert config.sft.lora.alpha == 8
            assert config.sft.lora.dropout == 0.0
            assert config.sft.lora.target_modules == "all-linear"
        finally:
            os.unlink(config_path)

    def test_access_null_optional_field(self) -> None:
        """Test accessing null optional field returns None."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "wandb": {
                "project": "CoACT",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.wandb.project == "CoACT"
        finally:
            os.unlink(config_path)


class TestConfigValidation:
    """Test configuration field validation."""

    def test_validate_agent_requires_model(self) -> None:
        """Test agent config must define a model."""
        config_data = {
            "agent": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(
                ConfigError,
                match="Missing required field: agent.model",
            ):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_validate_positive_integers(self) -> None:
        """Test positive integer validation."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
                "max_tokens": -100,  # Invalid: negative
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="must be positive"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_validate_float_range(self) -> None:
        """Test float range validation (e.g., temperature)."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
                "temperature": -1.0,  # Invalid: negative
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="must be non-negative"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_load_accepts_integer_yaml_scalars_for_agent_float_fields(self) -> None:
        """Test YAML integer scalars are accepted for agent float fields."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
                "temperature": 0,
                "top_p": 1,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.agent.temperature == 0
            assert config.agent.top_p == 1
        finally:
            os.unlink(config_path)

    def test_validate_non_empty_string(self) -> None:
        """Test non-empty string validation."""
        config_data = {
            "agent": {
                "model": "",  # Invalid: empty string
                "api_endpoint": "http://localhost:8000/v1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="non-empty string"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)

    def test_validate_list_not_empty(self) -> None:
        """Test list validation (must not be empty for required lists)."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": [],  # Invalid: empty list
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="non-empty list"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)


class TestConfigDefaults:
    """Test configuration default values."""

    def test_default_optional_fields(self) -> None:
        """Test that optional fields have sensible defaults."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            # Check default values
            assert config.sft.seed == 42
            assert config.sft.learning_rate == 2.0e-4
            assert config.sft.max_length == 8192
            assert config.sft.data_preparation.skip_compression_max_tokens == 256
            assert config.sft.dataloader_drop_last is True
        finally:
            os.unlink(config_path)


class TestConfigGet:
    """Test get method for flexible access."""

    def test_get_existing_field(self) -> None:
        """Test getting an existing field."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.get("agent.model") == "openai/Qwen3.5-35B-A3B-FP8"
            assert config.get("agent.api_endpoint") == "http://localhost:8000/v1"
        finally:
            os.unlink(config_path)

    def test_get_nonexistent_field_with_default(self) -> None:
        """Test getting non-existent field with default value."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            assert config.get("nonexistent.field", "default") == "default"
        finally:
            os.unlink(config_path)

    def test_get_nonexistent_field_no_default(self) -> None:
        """Test getting non-existent field without default raises error."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config.load(config_path)
            with pytest.raises(ConfigError, match="Configuration path not found"):
                config.get("nonexistent.field")
        finally:
            os.unlink(config_path)


class TestConfigSave:
    """Test configuration saving."""

    def test_save_config(self) -> None:
        """Test saving configuration to file."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix="_saved.yaml", delete=False) as f:
            save_path = f.name

        try:
            config = Config.load(config_path)
            config.save(save_path)

            # Verify saved file
            loaded = Config.load(save_path)
            assert loaded.agent.model == config.agent.model
            assert loaded.agent.api_key == config.agent.api_key
            assert loaded.sft.rollout.model == config.sft.rollout.model
            assert loaded.sft.rollout.api_key == config.sft.rollout.api_key
            assert (
                loaded.evaluation.coact.model_retry_stop_after_attempt
                == config.evaluation.coact.model_retry_stop_after_attempt
            )
            assert (
                loaded.sft.data_preparation.skip_compression_max_tokens
                == config.sft.data_preparation.skip_compression_max_tokens
            )
            assert loaded.sft.dataloader_drop_last == config.sft.dataloader_drop_last
            assert loaded.paths.eval_run_dir == config.paths.eval_run_dir
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_invalid_skip_compression_max_tokens(self) -> None:
        """Test skip_compression_max_tokens validation."""
        config_data = {
            "agent": {
                "model": "openai/Qwen3.5-35B-A3B-FP8",
                "api_endpoint": "http://localhost:8000/v1",
            },
            "sft": {
                "data_preparation": {
                    "skip_compression_max_tokens": 0,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ConfigError, match="skip_compression_max_tokens must be positive"):
                Config.load(config_path)
        finally:
            os.unlink(config_path)


class TestPromptTemplates:
    """Test prompt template loading and rendering."""

    def test_load_compression_prompt(self) -> None:
        """Test loading compression prompt."""
        from src.config.prompts import COMPRESSION_PROMPT

        prompt = COMPRESSION_PROMPT
        assert "compression" in prompt.lower()
        assert "{{ goal" in prompt
        assert "{{ context_focus_question }}" in prompt
        assert "{{ tool_output" in prompt
        assert "output it unchanged — do not compress or summarize code content" not in prompt

    def test_render_compression_prompt(self) -> None:
        """Test rendering compression prompt with variables."""
        from src.config.prompts import render_compression_prompt

        goal = "Fix the bug in user authentication"
        context_focus_question = "Check if the login function handles empty passwords"
        tool_output = "1> [Tool 1] returncode=0\n2> Error: Password cannot be empty at line 42"

        rendered = render_compression_prompt(goal, context_focus_question, tool_output)
        assert goal in rendered
        assert context_focus_question in rendered
        assert tool_output in rendered
        assert '{"type":"unchanged","content":null}' in rendered

    def test_render_compression_prompt_with_special_chars(self) -> None:
        """Test rendering compression prompt with special characters."""
        from src.config.prompts import render_compression_prompt

        goal = "Fix bug with $100 deposit"
        context_focus_question = "Check for SQL injection in 'user input'"
        tool_output = "Error: ' OR 1=1 -- detected"

        rendered = render_compression_prompt(goal, context_focus_question, tool_output)
        assert "$100" in rendered
        assert "' OR 1=1 --" in rendered

    def test_render_compression_prompt_with_unicode(self) -> None:
        """Test rendering compression prompt with Unicode characters."""
        from src.config.prompts import render_compression_prompt

        goal = "Fix bug in emoji handling"
        context_focus_question = "Check Unicode support for 日本語 and emojis 😊"
        tool_output = "Error: UnicodeEncodeError at position 100"

        rendered = render_compression_prompt(goal, context_focus_question, tool_output)
        assert "日本語" in rendered
        assert "😊" in rendered

    def test_render_compression_prompt_with_empty_values(self) -> None:
        """Test rendering compression prompt with empty values."""
        from src.config.prompts import render_compression_prompt

        rendered = render_compression_prompt("", "", "")
        assert "{goal}" not in rendered  # Should be replaced
        assert "{context_focus_question}" not in rendered
        assert "{tool_output}" not in rendered

    def test_render_compression_prompt_with_null_values(self) -> None:
        """Test rendering compression prompt with None values."""
        from src.config.prompts import render_compression_prompt

        rendered = render_compression_prompt(None, None, None)  # type: ignore
        # Should handle None gracefully
        assert isinstance(rendered, str)

    def test_render_compression_prompt_with_very_long_output(self) -> None:
        """Test rendering compression prompt with very long tool output."""
        from src.config.prompts import render_compression_prompt

        tool_output = "Error: " + "x" * 10000  # Very long output
        rendered = render_compression_prompt("Goal", "Intent", tool_output)
        assert len(rendered) > 10000
        assert tool_output in rendered

    def test_render_compression_prompt_code_example_is_compressed(self) -> None:
        """Test that the prompt no longer includes a code passthrough shortcut."""
        from src.config.prompts import render_compression_prompt

        prompt = render_compression_prompt("Goal", "Intent", "1> [Tool 1] returncode=0\n2> Output")
        assert '{"type":"unchanged","content":null}' in prompt
        assert "output it unchanged — do not compress or summarize code content" not in prompt
        assert "Source code from cat command (keep unchanged)" not in prompt

    def test_render_compression_prompt_documents_json_cases(self) -> None:
        """Test that the prompt documents the JSON response cases."""
        from src.config.prompts import render_compression_prompt

        prompt = render_compression_prompt(
            "Goal",
            "Inspect the relevant code path",
            "1> [Tool 1] returncode=0\n2> def foo():\n3>     return 1",
        )

        assert "Return EXACTLY one JSON object with fields in this order:" in prompt
        assert "Global rules:" in prompt
        assert "## Context Focus Question" in prompt
        assert "## Tool Call Executed" in prompt
        assert "Choose exactly one case:" in prompt
        assert "Case 1: No compression needed" in prompt
        assert "Case 2: Plain-text compression" in prompt
        assert "Case 3: Code-snippet compression" in prompt
        assert '- `"content"` must be `null`.' in prompt
        assert ('- `"content"` must contain only the compressed tool output content.') in prompt
        assert "- Do not include line numbers like `1>` or markdown code fences." in prompt
        assert (
            '- `"content"` must be a non-empty JSON array of sorted, non-overlapping omit '
            "ranges within Original Tool Output."
        ) in prompt
        assert (
            '- Entry format: `"N:summary"` or `"N-M:summary"`. Lines not covered by omit '
            "ranges are kept verbatim for the agent."
        ) in prompt
        assert (
            "Keep enclosing function, class, and control-flow structure needed to understand "
            "any kept line."
        ) in prompt
        assert '"type"' in prompt
        assert '"content"' in prompt
        assert '{"type":"unchanged","content":null}' in prompt
        assert (
            '{"type":"plain","content":"All selected tests passed; no failure traceback '
            'was present."}'
        ) in prompt
        assert (
            '{"type":"code","content":["1-5:license header and imports",'
            '"40-58:unrelated helper function"]}'
        ) in prompt
        assert "No answer" not in prompt
        assert "压缩了" not in prompt
        assert "FORMAT:" not in prompt
        assert "### Step " not in prompt
        assert "\nRules:\n" not in prompt
