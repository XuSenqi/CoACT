"""Mini-SWE-Agent runner wrapper.

This module provides a wrapper for running mini-swe-agent programmatically
with custom configuration for vLLM endpoints and trajectory collection.

Supports both local and Docker environments for running SWE-bench instances.
"""

import copy
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.utils.serialize import recursive_merge

from src.agent.context_focus_litellm_model import ContextFocusLitellmModel
from src.agent.trajectory_parser import extract_bash_command_from_query
from src.config.config import Config
from src.config.prompts import (
    COMPRESSION_NOTICE_GUIDANCE,
    CONTEXT_FOCUS_GUIDANCE,
    OBSERVATION_TEMPLATE,
    TRAJECTORY_COLLECTION_NOTE,
)

logger = logging.getLogger(__name__)


DEFAULT_RESPONSE_REQUIREMENTS = "\n".join(
    [
        "1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish",
        "2. Provide one or more bash tool calls to execute",
    ]
)


def _inject_response_guidance(default_template: str, extra_lines: list[str]) -> str:
    """Insert extra per-response guidance after the default requirements block.

    Args:
        default_template: Default Mini-SWE-Agent instance template.
        extra_lines: Guidance lines to append after ``DEFAULT_RESPONSE_REQUIREMENTS``.

    Returns:
        The template with ``extra_lines`` inserted next to the per-response block.

    Raises:
        ValueError: If the expected per-response block is missing.
    """
    if DEFAULT_RESPONSE_REQUIREMENTS not in default_template:
        raise ValueError(
            "Mini-SWE-Agent instance template is missing the default response requirements"
        )

    replacement = f"{DEFAULT_RESPONSE_REQUIREMENTS}\n" + "\n".join(extra_lines)
    return default_template.replace(DEFAULT_RESPONSE_REQUIREMENTS, replacement, 1)


def build_enhanced_instance_template(
    default_template: str,
    *,
    trajectory_collection_mode: bool = False,
) -> str:
    """Add context-focus guidance to the default instance template.

    Used by CFQ-consuming strategies (CoACT/swepruner).

    Args:
        default_template: Default Mini-SWE-Agent instance template.
        trajectory_collection_mode: Whether the template is used for raw
            trajectory collection before compression is active.

    Returns:
        The default template with additional bash-tool guidance inserted next
        to the existing per-response instructions.

    Raises:
        ValueError: If the expected per-response block is missing.
    """
    extra_lines: list[str] = [CONTEXT_FOCUS_GUIDANCE]
    if trajectory_collection_mode:
        extra_lines.append(TRAJECTORY_COLLECTION_NOTE)

    return _inject_response_guidance(default_template, extra_lines)


def build_compression_notice_template(default_template: str) -> str:
    """Add a transparent-compression notice to the default instance template.

    Used by task-agnostic compressors (e.g. LLMLingua-2) that compress tool
    outputs unconditionally without the agent's involvement, so the agent is
    told its observations are compressed but is not asked for a focus question.

    Args:
        default_template: Default Mini-SWE-Agent instance template.

    Returns:
        The template with the compression notice inserted next to the
        per-response instructions.

    Raises:
        ValueError: If the expected per-response block is missing.
    """
    return _inject_response_guidance(default_template, [COMPRESSION_NOTICE_GUIDANCE])


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for MiniSWERunner."""

    model_name: str
    """Model name for LiteLLM."""

    api_base: str | None = None
    """Optional API base URL for OpenAI-compatible endpoints."""

    api_key: str | None = None
    """Optional API key passed only to this agent model."""

    step_limit: int = 0
    """Maximum steps (0 = unlimited)."""

    cost_limit: float = 0.0
    """Cost limit (0 = disabled for local models)."""

    output_path: str | None = None
    """Path to save trajectory."""

    # Inference parameters
    temperature: float = 0.6
    """Sampling temperature for deterministic tool calling."""

    top_p: float | None = None
    """Nucleus sampling parameter."""

    top_k: int | None = 20
    """Top-k sampling."""

    min_p: float | None = 0.0
    """Minimum probability for token sampling."""

    presence_penalty: float | None = 0.0
    """Presence penalty for encouraging new topics."""

    repetition_penalty: float | None = 1.0
    """Repetition penalty to avoid redundancy."""

    max_tokens: int = 81920
    """Maximum tokens in response."""

    request_timeout: float = 300.0
    """Per-request LiteLLM timeout in seconds."""

    model_retry_stop_after_attempt: int = 3
    """Maximum model retry attempts inside mini-swe-agent."""

    # Docker configuration
    use_docker: bool = True
    """Whether to use Docker environment for SWE instances."""

    docker_timeout: str = "2h"
    """Timeout for Docker container (default: 2 hours)."""

    trajectory_collection_mode: bool = False
    """Whether to add collection-only guidance about inactive compression."""

    cfq_enabled: bool = True
    """Expose ``context_focus_question`` to the agent via the bash tool schema
    and the instance-template guidance. Disable for strategies that operate on
    a vanilla agent (VanillaCompression, SlidingWindow, AgentDiet)."""

    compression_notice_enabled: bool = False
    """Inject a transparent-compression notice into the instance template for
    task-agnostic compressors (e.g. LLMLingua-2) that compress tool outputs
    unconditionally. Ignored when ``cfq_enabled`` is True, since the CFQ
    guidance already tells the agent its outputs are compressed."""


def create_runner_config_from_config(config: Config) -> RunnerConfig:
    """Create RunnerConfig from the unified Config.

    Args:
        config: The unified configuration object

    Returns:
        RunnerConfig for mini-swe-agent
    """
    return RunnerConfig(
        model_name=config.agent.model,
        api_base=config.agent.api_endpoint,
        api_key=config.agent.api_key,
        step_limit=config.agent.step_limit,
        cost_limit=0.0,
        output_path=str(Path(config.paths.trajectory_dir) / "trajectories"),
        # Inference parameters from config
        temperature=config.agent.temperature,
        top_p=config.agent.top_p,
        top_k=config.agent.top_k,
        min_p=config.agent.min_p,
        presence_penalty=config.agent.presence_penalty,
        repetition_penalty=config.agent.repetition_penalty,
        max_tokens=config.agent.max_tokens,
        request_timeout=config.agent.request_timeout,
        model_retry_stop_after_attempt=config.agent.model_retry_stop_after_attempt,
        use_docker=True,
    )


class MiniSWERunner:
    """Wrapper for mini-swe-agent with vLLM integration.

    This class wraps the mini-swe-agent DefaultAgent to:
    - Configure vLLM endpoint
    - Add bash `context_focus_question` guidance to the instance prompt
    - Collect and return trajectories
    - Support both local and Docker environments

    Configuration Strategy:
    - Loads default swebench.yaml config from mini-swe-agent
    - Augments the default instance_template with extra bash-tool guidance
    - Uses default system_template from mini-swe-agent
    - Creates Docker environment for each instance with proper image
    """

    def __init__(self, config: RunnerConfig) -> None:
        """Initialize the runner.

        Args:
            config: Runner configuration
        """
        self.config = config
        self._agent: Any | None = None
        self._model: Any | None = None
        self._env: Any | None = None
        self._lock = threading.Lock()

    def _create_docker_environment(self, image_name: str) -> Any:
        """Create a Docker environment for the given image.

        Args:
            image_name: Docker image name (e.g., swebench/swesmith.x86_64.xxx)

        Returns:
            DockerEnvironment instance
        """
        from minisweagent.environments.docker import DockerEnvironment

        return DockerEnvironment(
            image=image_name,
            cwd="/testbed",
            timeout=60,  # Per-command timeout
            container_timeout=self.config.docker_timeout,  # Container lifetime
        )

    def _create_local_environment(self) -> Any:
        """Create a local environment.

        Returns:
            LocalEnvironment instance
        """
        from minisweagent.environments.local import LocalEnvironment

        return LocalEnvironment()

    def _cleanup_environment(self) -> None:
        """Clean up the current environment."""
        if self._env is not None:
            try:
                # DockerEnvironment has a cleanup method
                if hasattr(self._env, "cleanup"):
                    logger.info("Cleaning up Docker environment")
                    self._env.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up environment: {e}")
            finally:
                self._env = None
                self._agent = None

    def _lazy_init_model(self) -> None:
        """Initialize just the model component thread-safely."""
        with self._lock:
            if self._model is not None:
                return

            # Load default swebench.yaml config
            default_config = get_config_from_spec(builtin_config_dir / "benchmarks/swebench.yaml")

            # Merge with our custom templates
            custom_model_config = {
                "observation_template": OBSERVATION_TEMPLATE,  # No length filtering
            }

            merged_config = recursive_merge(
                default_config,
                {"model": custom_model_config},
            )

            # Create model with vLLM configuration
            merged_model_kwargs: dict[str, Any] = {
                k: v
                for k, v in {
                    **merged_config.get("model", {}).get("model_kwargs", {}),
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                    "presence_penalty": self.config.presence_penalty,
                    "repetition_penalty": self.config.repetition_penalty,
                    "max_tokens": self.config.max_tokens,
                    "timeout": self.config.request_timeout,
                }.items()
                if v is not None
            }
            if self.config.api_base is not None:
                merged_model_kwargs["api_base"] = self.config.api_base
            if self.config.api_key is not None:
                merged_model_kwargs["api_key"] = self.config.api_key

            model_cls: type = ContextFocusLitellmModel if self.config.cfq_enabled else LitellmModel
            self._model = model_cls(
                model_name=self.config.model_name,
                model_kwargs=merged_model_kwargs,
                cost_tracking="ignore_errors",
                observation_template=merged_config.get("model", {}).get("observation_template"),
                format_error_template=merged_config.get("model", {}).get("format_error_template"),
            )

    def _lazy_init(
        self,
        image_name: str | None = None,
        agent_class: type = DefaultAgent,
        agent_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the agent components.

        Args:
            image_name: Docker image name for the environment
            agent_class: Agent class to instantiate (default: DefaultAgent)
            agent_kwargs: Additional kwargs to pass to agent constructor
        """
        # 1. Thread-safe model initialization
        self._lazy_init_model()

        # 2. Environment clean up and re-init (Stateful)
        self._cleanup_environment()

        # 3. Prepare Agent-specific configuration
        default_config = get_config_from_spec(builtin_config_dir / "benchmarks/swebench.yaml")

        os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = str(
            self.config.model_retry_stop_after_attempt
        )

        instance_template = default_config["agent"]["instance_template"]
        if self.config.cfq_enabled:
            instance_template = build_enhanced_instance_template(
                instance_template,
                trajectory_collection_mode=self.config.trajectory_collection_mode,
            )
        elif self.config.compression_notice_enabled:
            instance_template = build_compression_notice_template(instance_template)

        custom_agent_config = {
            "cost_limit": self.config.cost_limit,
            "step_limit": self.config.step_limit,
            "instance_template": instance_template,
        }

        # Merge defaults with agent overrides
        merged_config = recursive_merge(
            default_config,
            {"agent": custom_agent_config},
        )

        # 4. Create environment
        if self.config.use_docker and image_name:
            logger.info(f"Creating Docker environment with image: {image_name}")
            self._env = self._create_docker_environment(image_name)
            self._current_image = image_name
        else:
            logger.info("Creating local environment")
            self._env = self._create_local_environment()
            self._current_image = None

        # 5. Create agent using the specified agent_class
        agent_init_kwargs = {
            "model": self._model,
            "env": self._env,
            **merged_config.get("agent", {}),
        }
        if agent_kwargs:
            agent_init_kwargs.update(agent_kwargs)
        self._agent = agent_class(**agent_init_kwargs)

    def run(
        self,
        task: str,
        instance_id: str | None = None,
        image_name: str | None = None,
        agent_class: type = DefaultAgent,
        agent_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the agent on a task.

        Args:
            task: The task/issue description to solve
            instance_id: Optional instance identifier
            image_name: Optional Docker image name for the environment
            agent_class: Agent class to instantiate (default: DefaultAgent)
            agent_kwargs: Additional kwargs to pass to agent constructor

        Returns:
            Dictionary with:
                - exit_status: The agent's exit status
                - submission: The final submission/result
                - trajectory: The full trajectory data
        """
        # Initialize if needed (creates environment and agent)
        self._lazy_init(image_name=image_name, agent_class=agent_class, agent_kwargs=agent_kwargs)

        # Run the agent
        result = self._agent.run(task)  # type: ignore

        # Get trajectory data
        trajectory_data = self._agent.serialize()  # type: ignore

        # Add instance_id if provided
        if instance_id and "info" in trajectory_data:
            trajectory_data["info"]["instance_id"] = instance_id

        return {
            "exit_status": result.get("exit_status", "unknown"),
            "submission": result.get("submission"),
            "trajectory": trajectory_data,
        }

    def save_trajectory(self, trajectory_data: dict[str, Any], path: str) -> None:
        """Save trajectory to a file.

        Args:
            trajectory_data: The trajectory data to save
            path: Path to save to (JSON format)
        """
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def annotate_compressed_output(
        compressed_output: str,
        original_output: str,
    ) -> str:
        """Format compressed output with a hint if compression happened.

        Args:
            compressed_output: The compressed tool output.
            original_output: The original tool output for comparison.

        Returns:
            Formatted output string. If compression happened, prepends a hint.
        """
        if compressed_output.strip() == original_output.strip():
            # No compression, return as-is
            return compressed_output
        else:
            # Compression happened, add hint
            return f"[Original tool output has been compressed]\n{compressed_output}"

    def continue_from_step(
        self,
        trajectory_messages: list[dict],
        step_index: int,
        compressed_output: str,
        returncode: int = 0,
        exception_info: str | None = None,
        use_original_output: bool = False,
    ) -> str | None:
        """Continue reasoning from a trajectory step with compressed tool output.

        This method allows injecting a compressed tool output into an existing
        trajectory and having the agent continue reasoning from that point.

        Args:
            trajectory_messages: The original trajectory's message list.
            step_index: The index of the assistant message whose tool output should be replaced.
            compressed_output: The compressed version of the first tool output.
            returncode: The return code to use for the compressed output (default: 0).
            exception_info: Exception information if any (default: None).
            use_original_output: Whether to continue with the untouched original
                tool message instead of replacing it with compressed text.

        Returns:
            The bash command the agent would execute next, or None if failed.

        Example:
            >>> runner = MiniSWERunner(config)
            >>> messages = trajectory_json["messages"]
            >>> # step_index points to the assistant message
            >>> next_cmd = runner.continue_from_step(
            ...     trajectory_messages=messages,
            ...     step_index=2,  # assistant message at index 2
            ...     compressed_output="File contents: main.py has function foo()",
            ...     returncode=0,
            ...     exception_info=None,
            ... )
        """
        self._lazy_init_model()

        # Verify step_index points to an assistant message
        if step_index >= len(trajectory_messages):
            logger.error(
                f"Step index {step_index} out of range "
                f"(trajectory has {len(trajectory_messages)} messages)"
            )
            return None

        assistant_msg = trajectory_messages[step_index]
        if assistant_msg.get("role") != "assistant":
            logger.error(
                f"Message at index {step_index} is not an assistant message, "
                f"got role: {assistant_msg.get('role')}"
            )
            return None

        tool_call_ids = [tool_call["id"] for tool_call in assistant_msg.get("tool_calls", [])]
        if len(tool_call_ids) != 1:
            logger.error(
                f"Assistant message at index {step_index} must have exactly one tool call, "
                f"got {len(tool_call_ids)}"
            )
            return None

        tool_msg_index = step_index + 1
        if tool_msg_index >= len(trajectory_messages):
            logger.error(f"No tool message found after assistant message at index {step_index}")
            return None

        tool_message = trajectory_messages[tool_msg_index]
        if tool_message.get("role") != "tool":
            logger.error(
                f"Expected a tool message immediately after assistant message at index "
                f"{step_index}, got role: {tool_message.get('role')}"
            )
            return None

        if tool_message.get("tool_call_id") != tool_call_ids[0]:
            logger.error(
                f"Tool messages after assistant step {step_index} do not match tool calls: "
                f"expected id {tool_call_ids[0]}, got {tool_message.get('tool_call_id')}"
            )
            return None

        next_index = tool_msg_index + 1
        if (
            next_index < len(trajectory_messages)
            and trajectory_messages[next_index].get("role") == "tool"
        ):
            logger.error(f"Unexpected additional tool message after step {step_index}")
            return None

        messages_copy = copy.deepcopy(trajectory_messages[:next_index])

        # Get original tool output and prepare the message content seen by the agent.
        original_output = trajectory_messages[tool_msg_index].get("content", "")
        if use_original_output:
            formatted_output = original_output
            output_label = "original tool output"
            output_length = len(original_output)
        else:
            formatted_output = self.annotate_compressed_output(compressed_output, original_output)
            output_label = "compressed output"
            output_length = len(compressed_output)

        # Replace the tool message content with formatted output
        messages_copy[tool_msg_index]["content"] = formatted_output

        # Update extra fields
        if not use_original_output and "extra" in messages_copy[tool_msg_index]:
            messages_copy[tool_msg_index]["extra"]["raw_output"] = compressed_output
            messages_copy[tool_msg_index]["extra"]["returncode"] = returncode
            messages_copy[tool_msg_index]["extra"]["exception_info"] = exception_info

        logger.info(
            f"Continuing from assistant step {step_index} (tool at {tool_msg_index}) "
            f"with {output_label} (length: {output_length})"
        )

        try:
            from minisweagent.exceptions import FormatError
        except ImportError:
            FormatError = None  # type: ignore

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Query the model directly instead of using self._agent.query()
                message = self._model.query(messages_copy)

                command = extract_bash_command_from_query(message)
                if command is not None:
                    return command

                # No tool calls - this might be a format issue
                logger.warning(
                    f"No bash tool calls in response (attempt {attempt + 1}/{max_retries})"
                )

            except Exception as e:
                # Handle FormatError by adding error message and retrying
                if FormatError and isinstance(e, FormatError):
                    logger.warning(f"FormatError on attempt {attempt + 1}/{max_retries}: {e}")
                    messages_copy.extend(e.messages)
                    continue
                else:
                    logger.error(f"Error during continuation: {e}")
                    return None

        logger.warning("Could not extract bash command after max retries")
        return None
