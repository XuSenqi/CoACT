"""Tests for Mini-SWE-Agent runner wrapper."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.miniswe_runner import (
    MiniSWERunner,
    RunnerConfig,
    build_compression_notice_template,
    build_enhanced_instance_template,
    create_runner_config_from_config,
)
from src.config.config import AgentConfig, Config, PathsConfig

# Check if minisweagent is installed
MINISWEAGENT_INSTALLED = importlib.util.find_spec("minisweagent") is not None


class TestRunnerConfig:
    """Tests for RunnerConfig dataclass."""

    def test_runner_config_creation(self) -> None:
        """Test creating a RunnerConfig."""
        config = RunnerConfig(
            model_name="openai/Qwen/Qwen3.5-35B",
            api_base="http://localhost:8000/v1",
            api_key="agent-key",
        )

        assert config.model_name == "openai/Qwen/Qwen3.5-35B"
        assert config.api_base == "http://localhost:8000/v1"
        assert config.api_key == "agent-key"
        assert config.step_limit == 0
        assert config.cost_limit == 0.0

    def test_runner_config_frozen(self) -> None:
        """Test that RunnerConfig is immutable."""
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )

        with pytest.raises(AttributeError):
            config.model_name = "modified"  # type: ignore


class TestCreateRunnerConfigFromConfig:
    """Tests for create_runner_config_from_config function."""

    def test_create_runner_config_from_unified_config(self) -> None:
        """Test creating RunnerConfig from unified Config."""
        agent_config = AgentConfig(
            model="openai/Qwen3.5-35B-A3B-FP8",
            api_endpoint="http://localhost:8000/v1",
            api_key="agent-key",
            step_limit=42,
            request_timeout=300.0,
            model_retry_stop_after_attempt=3,
        )
        paths_config = PathsConfig(
            trajectory_dir="data/trajectories",
        )

        config = MagicMock(spec=Config)
        config.agent = agent_config
        config.paths = paths_config

        runner_config = create_runner_config_from_config(config)

        assert runner_config.model_name == "openai/Qwen3.5-35B-A3B-FP8"
        assert runner_config.api_base == "http://localhost:8000/v1"
        assert runner_config.api_key == "agent-key"
        assert runner_config.cost_limit == 0.0
        assert runner_config.step_limit == 42
        assert runner_config.request_timeout == 300.0
        assert runner_config.model_retry_stop_after_attempt == 3

    def test_create_runner_config_from_provider_model(self) -> None:
        """Test creating RunnerConfig for a direct LiteLLM provider model."""
        agent_config = AgentConfig(
            model="openai/gpt-4o-mini",
            api_endpoint=None,
            step_limit=7,
        )
        paths_config = PathsConfig(
            trajectory_dir="data/trajectories",
        )

        config = MagicMock(spec=Config)
        config.agent = agent_config
        config.paths = paths_config

        runner_config = create_runner_config_from_config(config)

        assert runner_config.model_name == "openai/gpt-4o-mini"
        assert runner_config.api_base is None
        assert runner_config.step_limit == 7


class TestMiniSWERunner:
    """Tests for MiniSWERunner class."""

    def test_runner_creation(self) -> None:
        """Test creating a MiniSWERunner."""
        config = RunnerConfig(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
        )

        runner = MiniSWERunner(config)

        assert runner.config == config
        assert runner._agent is None  # Lazy init

    @patch("src.agent.miniswe_runner.MiniSWERunner._lazy_init")
    def test_run_returns_result(self, mock_lazy_init: MagicMock) -> None:
        """Test run method returns expected result."""
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )

        runner = MiniSWERunner(config)

        # Mock the agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"exit_status": "submitted", "submission": "Fixed!"}
        mock_agent.serialize.return_value = {"info": {}, "messages": []}
        runner._agent = mock_agent

        result = runner.run("Fix the bug")

        assert result["exit_status"] == "submitted"
        assert result["submission"] == "Fixed!"
        assert "trajectory" in result

    @patch("src.agent.miniswe_runner.MiniSWERunner._lazy_init")
    def test_run_with_instance_id(self, mock_lazy_init: MagicMock) -> None:
        """Test run with instance_id adds it to trajectory."""
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )

        runner = MiniSWERunner(config)

        mock_agent = MagicMock()
        mock_agent.run.return_value = {"exit_status": "submitted"}
        mock_agent.serialize.return_value = {"info": {}}
        runner._agent = mock_agent

        result = runner.run("Task", instance_id="test-001")

        assert result["trajectory"]["info"]["instance_id"] == "test-001"

    def test_save_trajectory(self, tmp_path: Path) -> None:
        """Test saving trajectory to file."""
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )

        runner = MiniSWERunner(config)

        trajectory_data = {"info": {"exit_status": "submitted"}, "messages": []}
        save_path = str(tmp_path / "trajectory.json")

        runner.save_trajectory(trajectory_data, save_path)

        assert Path(save_path).exists()

        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded == trajectory_data

    @patch("src.agent.miniswe_runner.MiniSWERunner._lazy_init_model")
    def test_continue_from_step_can_keep_original_tool_message(
        self,
        mock_lazy_init_model: MagicMock,
    ) -> None:
        """Test continuing from a step without replacing the original tool message."""
        del mock_lazy_init_model
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )
        runner = MiniSWERunner(config)
        runner._model = MagicMock()
        runner._model.query.return_value = {"tool_calls": []}

        trajectory_messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "assistant",
                "content": "Inspect file",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "bash"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"returncode": 0, "output": "file contents"}',
                "extra": {
                    "raw_output": "file contents",
                    "returncode": 0,
                    "exception_info": None,
                },
            },
        ]

        runner.continue_from_step(
            trajectory_messages=trajectory_messages,
            step_index=1,
            compressed_output="<KEEP_ORIGINAL>",
            use_original_output=True,
        )

        query_messages = runner._model.query.call_args.args[0]
        assert query_messages[2]["content"] == trajectory_messages[2]["content"]
        assert query_messages[2]["extra"]["raw_output"] == "file contents"

    @patch("src.agent.miniswe_runner.MiniSWERunner._lazy_init_model")
    def test_continue_from_step_rejects_multi_tool_call_steps(
        self,
        mock_lazy_init_model: MagicMock,
    ) -> None:
        """Continuation only supports single tool-call compression targets."""
        del mock_lazy_init_model
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )
        runner = MiniSWERunner(config)
        runner._model = MagicMock()

        trajectory_messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "bash"}},
                    {"id": "call_2", "type": "function", "function": {"name": "bash"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"returncode": 0, "output": "first"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": '{"returncode": 0, "output": "second"}',
            },
        ]

        command = runner.continue_from_step(
            trajectory_messages=trajectory_messages,
            step_index=1,
            compressed_output="compressed aggregate",
        )

        assert command is None
        runner._model.query.assert_not_called()


class TestMiniSWERunnerLazyInit:
    """Tests for lazy initialization."""

    def test_lazy_init_creates_components(self) -> None:
        """Test that lazy init creates model, env, and agent."""
        config = RunnerConfig(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
        )

        runner = MiniSWERunner(config)

        # Patch the imports
        with patch.dict(
            "sys.modules",
            {
                "minisweagent": MagicMock(),
                "minisweagent.config": MagicMock(),
                "minisweagent.agents": MagicMock(),
                "minisweagent.agents.default": MagicMock(),
                "minisweagent.environments": MagicMock(),
                "minisweagent.environments.local": MagicMock(),
                "minisweagent.models": MagicMock(),
                "minisweagent.models.litellm_model": MagicMock(),
                "minisweagent.utils": MagicMock(),
                "minisweagent.utils.serialize": MagicMock(),
            },
        ):
            # Call lazy init
            runner._lazy_init()

            # Check that components are created
            assert runner._model is not None
            assert runner._env is not None
            assert runner._agent is not None

    def test_lazy_init_only_once(self) -> None:
        """Test that lazy init of MODEL only runs once (thread-safe)."""
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )

        runner = MiniSWERunner(config)

        with patch.dict(
            "sys.modules",
            {
                "minisweagent": MagicMock(),
                "minisweagent.config": MagicMock(),
                "minisweagent.agents": MagicMock(),
                "minisweagent.agents.default": MagicMock(),
                "minisweagent.environments": MagicMock(),
                "minisweagent.environments.local": MagicMock(),
                "minisweagent.models": MagicMock(),
                "minisweagent.models.litellm_model": MagicMock(),
                "minisweagent.utils": MagicMock(),
                "minisweagent.utils.serialize": MagicMock(),
            },
        ):
            # Initialize model twice - should only create once
            runner._lazy_init_model()
            first_model = runner._model

            runner._lazy_init_model()
            assert runner._model is first_model  # Same instance

    @patch("src.agent.miniswe_runner.ContextFocusLitellmModel")
    def test_lazy_init_model_passes_api_key_to_litellm(
        self,
        mock_model_class: MagicMock,
    ) -> None:
        """Agent API keys should be scoped to the runner's LiteLLM model."""
        config = RunnerConfig(
            model_name="openai/test-agent",
            api_base="https://agent.example/v1",
            api_key="agent-key",
        )

        runner = MiniSWERunner(config)

        runner._lazy_init_model()

        model_kwargs = mock_model_class.call_args.kwargs["model_kwargs"]
        assert model_kwargs["api_base"] == "https://agent.example/v1"
        assert model_kwargs["api_key"] == "agent-key"

    @pytest.mark.skipif(
        MINISWEAGENT_INSTALLED, reason="minisweagent is installed, cannot test import error"
    )
    def test_lazy_init_import_error(self) -> None:
        """Test that missing minisweagent raises ImportError."""
        config = RunnerConfig(
            model_name="test",
            api_base="http://test",
        )

        runner = MiniSWERunner(config)

        # Clear any existing model
        runner._model = None

        # Mock the import to raise ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("minisweagent"):
                raise ImportError("No module named 'minisweagent'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                runner._lazy_init_model()

            # Should raise ImportError about minisweagent
            assert "minisweagent" in str(exc_info.value)


class TestDockerEnvironmentCleanup:
    """Tests for Docker environment cleanup."""

    def test_cleanup_called_before_new_environment(self) -> None:
        """Test that cleanup is called before creating a new environment."""
        config = RunnerConfig(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
        )

        runner = MiniSWERunner(config)

        # Mock old environment with cleanup method
        old_env = MagicMock()
        old_env.cleanup = MagicMock()
        runner._env = old_env

        with patch.dict(
            "sys.modules",
            {
                "minisweagent": MagicMock(),
                "minisweagent.config": MagicMock(),
                "minisweagent.agents": MagicMock(),
                "minisweagent.agents.default": MagicMock(),
                "minisweagent.environments": MagicMock(),
                "minisweagent.environments.local": MagicMock(),
                "minisweagent.environments.docker": MagicMock(),
                "minisweagent.models": MagicMock(),
                "minisweagent.models.litellm_model": MagicMock(),
                "minisweagent.utils": MagicMock(),
                "minisweagent.utils.serialize": MagicMock(),
            },
        ):
            runner._lazy_init("new_image")

        # Verify cleanup was called on old environment
        old_env.cleanup.assert_called_once()

    def test_cleanup_handles_errors(self) -> None:
        """Test that cleanup handles errors gracefully."""
        config = RunnerConfig(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
        )

        runner = MiniSWERunner(config)

        # Mock old environment that raises error on cleanup
        old_env = MagicMock()
        old_env.cleanup = MagicMock(side_effect=Exception("Cleanup failed"))
        runner._env = old_env

        # Should not raise, just log warning
        runner._cleanup_environment()

        # Verify cleanup was attempted
        old_env.cleanup.assert_called_once()

        # Verify state was reset despite error
        assert runner._env is None
        assert runner._agent is None

    def test_each_instance_gets_fresh_environment(self) -> None:
        """Test that each instance gets a fresh environment (cleanup called)."""
        config = RunnerConfig(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
        )

        runner = MiniSWERunner(config)

        with patch.dict(
            "sys.modules",
            {
                "minisweagent": MagicMock(),
                "minisweagent.config": MagicMock(),
                "minisweagent.agents": MagicMock(),
                "minisweagent.agents.default": MagicMock(),
                "minisweagent.environments": MagicMock(),
                "minisweagent.environments.local": MagicMock(),
                "minisweagent.environments.docker": MagicMock(),
                "minisweagent.models": MagicMock(),
                "minisweagent.models.litellm_model": MagicMock(),
                "minisweagent.utils": MagicMock(),
                "minisweagent.utils.serialize": MagicMock(),
            },
        ):
            # First instance
            runner._lazy_init("image_1")
            first_env = runner._env

            # Mock cleanup on first env
            first_env.cleanup = MagicMock()
            cleanup_mock = first_env.cleanup

            # Second instance - cleanup should be called on first env
            runner._lazy_init("image_2")

            # Verify cleanup was called before creating second environment
            cleanup_mock.assert_called_once()

    @patch("src.agent.miniswe_runner.DefaultAgent")
    @patch("src.agent.miniswe_runner.get_config_from_spec")
    @patch("src.agent.miniswe_runner.recursive_merge")
    @patch("src.agent.miniswe_runner.MiniSWERunner._create_local_environment")
    @patch("src.agent.miniswe_runner.MiniSWERunner._lazy_init_model")
    def test_lazy_init_adds_context_focus_guidance(
        self,
        mock_lazy_init_model: MagicMock,
        mock_create_local_environment: MagicMock,
        mock_recursive_merge: MagicMock,
        mock_get_config_from_spec: MagicMock,
        mock_default_agent: MagicMock,
    ) -> None:
        """Test lazy init keeps the default template unchanged outside collection mode."""
        del mock_default_agent
        del mock_lazy_init_model
        config = RunnerConfig(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
            use_docker=False,
        )
        runner = MiniSWERunner(config)
        runner._model = MagicMock()

        default_template = (
            "Header\n\n"
            "For each response:\n\n"
            "1. Include a THOUGHT section explaining your reasoning and what you're trying to "
            "accomplish\n"
            "2. Provide one or more bash tool calls to execute\n"
            "\n"
            "## Important Boundaries\n"
        )
        default_config = {
            "agent": {
                "instance_template": default_template,
                "system_template": "system",
            },
        }
        mock_get_config_from_spec.return_value = default_config
        mock_recursive_merge.side_effect = lambda left, right: {
            **left,
            "agent": {
                **left.get("agent", {}),
                **right.get("agent", {}),
            },
        }
        mock_create_local_environment.return_value = MagicMock()

        runner._lazy_init()

        merged_agent_config = mock_recursive_merge.call_args.args[1]["agent"]
        instance_template = merged_agent_config["instance_template"]
        assert "For each response:" in instance_template
        assert "## Important Boundaries" in instance_template
        assert (
            "1. Include a THOUGHT section explaining your reasoning and what you're trying "
            "to accomplish"
        ) in instance_template
        assert "2. Provide one or more bash tool calls to execute" in instance_template
        assert "will not trigger compression yet" not in instance_template

    def test_build_enhanced_instance_template_preserves_default_content(self) -> None:
        """Test non-collection enhancement adds context-focus guidance."""
        default_template = (
            "<instructions>\n"
            "For each response:\n\n"
            "1. Include a THOUGHT section explaining your reasoning and what you're trying to "
            "accomplish\n"
            "2. Provide one or more bash tool calls to execute\n"
            "\n"
            "## Important Boundaries\n"
            "</instructions>\n"
        )

        enhanced_template = build_enhanced_instance_template(default_template)

        assert "context_focus_question" in enhanced_template
        assert "will not trigger compression yet" not in enhanced_template
        assert "## Important Boundaries" in enhanced_template
        assert (
            "1. Include a THOUGHT section explaining your reasoning and what you're trying "
            "to accomplish"
        ) in enhanced_template
        assert "2. Provide one or more bash tool calls to execute" in enhanced_template

    def test_build_enhanced_instance_template_collection_mode_adds_note(self) -> None:
        """Collection mode should add the training-only note for focus questions."""
        default_template = (
            "<instructions>\n"
            "For each response:\n\n"
            "1. Include a THOUGHT section explaining your reasoning and what you're trying to "
            "accomplish\n"
            "2. Provide one or more bash tool calls to execute\n"
            "\n"
            "## Important Boundaries\n"
            "</instructions>\n"
        )

        enhanced_template = build_enhanced_instance_template(
            default_template,
            trajectory_collection_mode=True,
        )

        assert "During trajectory collection" in enhanced_template
        assert "will not trigger compression yet" in enhanced_template

    def test_build_compression_notice_template_adds_transparent_notice(self) -> None:
        """Task-agnostic notice tells the agent outputs are compressed, without CFQ."""
        default_template = (
            "<instructions>\n"
            "For each response:\n\n"
            "1. Include a THOUGHT section explaining your reasoning and what you're trying to "
            "accomplish\n"
            "2. Provide one or more bash tool calls to execute\n"
            "\n"
            "## Important Boundaries\n"
            "</instructions>\n"
        )

        notice_template = build_compression_notice_template(default_template)

        assert "automatically compressed" in notice_template
        # Must NOT ask for a context_focus_question (LLMLingua-2 ignores it).
        assert "context_focus_question" not in notice_template
        assert "## Important Boundaries" in notice_template
        assert "2. Provide one or more bash tool calls to execute" in notice_template
