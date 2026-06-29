"""Tests for trajectory parsing from mini-swe-agent outputs."""

import pytest

from src.agent.trajectory_parser import (
    ToolOutput,
    Trajectory,
    TrajectoryStep,
    parse_trajectory,
)


class TestToolOutputDataclass:
    """Tests for ToolOutput dataclass."""

    def test_tool_output_creation(self) -> None:
        """Test creating a ToolOutput."""
        output = ToolOutput(
            returncode=0,
            output="file contents",
        )

        assert output.returncode == 0
        assert output.output == "file contents"
        assert output.exception is None

    def test_tool_output_with_exception(self) -> None:
        """Test ToolOutput with exception info."""
        output = ToolOutput(
            returncode=1,
            output="error",
            exception="FileNotFoundError: file.py",
        )

        assert output.returncode == 1
        assert output.exception == "FileNotFoundError: file.py"

    def test_tool_output_frozen(self) -> None:
        """Test that ToolOutput is immutable."""
        output = ToolOutput(returncode=0, output="test")

        with pytest.raises(AttributeError):
            output.returncode = 1  # type: ignore


class TestTrajectoryStepDataclass:
    """Tests for TrajectoryStep dataclass."""

    def test_trajectory_step_creation(self) -> None:
        """Test creating a TrajectoryStep."""
        tool_output = (ToolOutput(returncode=0, output="file contents"),)
        step = TrajectoryStep(
            goal="Fix the bug",
            tool_output=tool_output,
            role="assistant",
            content="I'll check the file",
            reasoning="I need to check the file contents first",
            tool_call=("cat file.py",),
            context_focus_question=("Inspect the file contents relevant to the bug.",),
        )

        assert step.goal == "Fix the bug"
        assert step.reasoning == "I need to check the file contents first"
        assert step.tool_output == tool_output
        assert step.tool_call == ("cat file.py",)
        assert step.context_focus_question == (
            "Inspect the file contents relevant to the bug.",
        )
        assert step.role == "assistant"
        assert step.content == "I'll check the file"

    def test_trajectory_step_frozen(self) -> None:
        """Test that TrajectoryStep is immutable."""
        step = TrajectoryStep(
            goal="Test",
            reasoning="Test reasoning",
            tool_output=(),
            tool_call=(),
        )

        with pytest.raises(AttributeError):
            step.goal = "modified"  # type: ignore

    def test_trajectory_step_optional_reasoning(self) -> None:
        """Test that reasoning can be None."""
        step = TrajectoryStep(
            goal="Test",
            reasoning=None,
            tool_output=(),
            tool_call=(),
        )

        assert step.reasoning is None

    def test_trajectory_step_empty_tool_call(self) -> None:
        """Test that tool_call can be empty tuple (last step)."""
        step = TrajectoryStep(
            goal="Test",
            reasoning="Final step",
            tool_output=(),
            tool_call=(),
        )

        assert step.tool_call == ()

    def test_trajectory_step_multiple_tool_outputs(self) -> None:
        """Test step with multiple tool outputs."""
        tool_outputs = (
            ToolOutput(returncode=0, output="first output"),
            ToolOutput(returncode=0, output="second output"),
        )
        step = TrajectoryStep(
            goal="Test",
            tool_output=tool_outputs,
            tool_call=(),
        )

        assert len(step.tool_output) == 2
        assert step.tool_output[0].output == "first output"

    def test_trajectory_step_multiple_tool_calls(self) -> None:
        """Test step with multiple bash commands."""
        step = TrajectoryStep(
            goal="Test",
            tool_output=(),
            tool_call=("ls -la", "cat file.py", "grep pattern"),
            context_focus_question=(
                "Inspect the repository root.",
                "Read the implementation in file.py.",
                "Find pattern matches.",
            ),
        )

        assert len(step.tool_call) == 3
        assert step.tool_call[0] == "ls -la"
        assert step.context_focus_question[1] == "Read the implementation in file.py."


class TestTrajectoryDataclass:
    """Tests for Trajectory dataclass."""

    def test_trajectory_creation(self) -> None:
        """Test creating a Trajectory."""
        steps = (
            TrajectoryStep(
                goal="Fix bug",
                reasoning="I need to check the files first",
                tool_output=(ToolOutput(returncode=0, output="output1"),),
                tool_call=("edit file",),
            ),
            TrajectoryStep(
                goal="Fix bug",
                reasoning="Now I'll verify the fix",
                tool_output=(ToolOutput(returncode=0, output="output2"),),
                tool_call=(),
            ),
        )

        trajectory = Trajectory(
            instance_id="test-001",
            issue_description="Fix the login bug",
            steps=steps,
            final_result="Fixed the bug",
            exit_status="submitted",
        )

        assert trajectory.instance_id == "test-001"
        assert len(trajectory.steps) == 2
        assert trajectory.final_result == "Fixed the bug"
        assert trajectory.exit_status == "submitted"

    def test_trajectory_frozen(self) -> None:
        """Test that Trajectory is immutable."""
        trajectory = Trajectory(
            instance_id="test",
            issue_description="Test",
            steps=(),
            final_result=None,
        )

        with pytest.raises(AttributeError):
            trajectory.instance_id = "modified"  # type: ignore

    def test_trajectory_default_exit_status(self) -> None:
        """Test default exit_status value."""
        trajectory = Trajectory(
            instance_id="test",
            issue_description="Test",
            steps=(),
            final_result=None,
        )

        assert trajectory.exit_status == "unknown"


class TestParseTrajectory:
    """Tests for parse_trajectory function."""

    def test_parse_trajectory_basic(self) -> None:
        """Test parsing a basic trajectory."""
        trajectory_json = {
            "info": {
                "exit_status": "submitted",
                "submission": "Fixed the bug by editing file.py",
            },
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Please solve this issue: Fix the login bug",
                },
                {
                    "role": "assistant",
                    "content": "I'll check the files.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": (
                                    '{"command": "ls -la", '
                                    '"context_focus_question": "Inspect the repository root."}'
                                ),
                            },
                        }
                    ],
                    "reasoning_content": "I need to list files to find the bug location",
                },
                {
                    "role": "tool",
                    "content": '{"returncode": 0, "output": "file1.py file2.py"}',
                },
                {
                    "role": "assistant",
                    "content": "Found the files.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": (
                                    '{"command": "cat file1.py", '
                                    '"context_focus_question": "Read the implementation in file1.py."}'
                                ),
                            },
                        }
                    ],
                    "reasoning_content": "Now I will read the file to understand the code",
                },
                {
                    "role": "tool",
                    "content": '{"returncode": 0, "output": "hello world code"}',
                },
                {
                    "role": "exit",
                    "content": "submit",
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert result.instance_id == "unknown"  # No instance_id in basic format
        assert result.issue_description == "Please solve this issue: Fix the login bug"
        assert result.exit_status == "submitted"
        assert result.final_result == "Fixed the bug by editing file.py"
        assert len(result.steps) == 5  # system, user, 2x assistant, exit
        # Find the assistant step with reasoning
        assistant_steps = [s for s in result.steps if s.role == "assistant"]
        assert len(assistant_steps) == 2
        assert assistant_steps[0].reasoning == "I need to list files to find the bug location"
        assert assistant_steps[0].tool_call == ("ls -la",)
        assert assistant_steps[0].context_focus_question == ("Inspect the repository root.",)

    def test_parse_trajectory_with_instance_id(self) -> None:
        """Test parsing trajectory with instance_id."""
        trajectory_json = {
            "info": {
                "exit_status": "submitted",
                "submission": "Done",
                "instance_id": "swe-bench-001",
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Please solve: Test issue",
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert result.instance_id == "swe-bench-001"

    def test_parse_trajectory_empty_messages(self) -> None:
        """Test parsing trajectory with no tool calls."""
        trajectory_json = {
            "info": {
                "exit_status": "error",
                "submission": None,
            },
            "messages": [
                {
                    "role": "user",
                    "content": "Please solve: Empty issue",
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert len(result.steps) == 1
        assert result.steps[0].role == "user"
        assert result.final_result is None
        assert result.exit_status == "error"

    def test_parse_trajectory_missing_info(self) -> None:
        """Test parsing trajectory with missing info section."""
        trajectory_json = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test issue",
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert result.exit_status == "unknown"
        assert result.final_result is None

    def test_parse_trajectory_extracts_tool_outputs(self) -> None:
        """Test that tool outputs are correctly extracted as ToolOutput objects."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Check file",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
                },
                {
                    "role": "tool",
                    "content": '{"returncode": 0, "output": "file1.py file2.py"}',
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert len(result.steps) == 2
        assistant_step = result.steps[1]
        assert assistant_step.role == "assistant"
        assert len(assistant_step.tool_output) == 1
        assert assistant_step.tool_output[0].returncode == 0
        assert "file1.py" in assistant_step.tool_output[0].output

    def test_parse_trajectory_extracts_tool_calls(self) -> None:
        """Test that next actions are correctly extracted as tuple."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Step 1",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
                },
                {"role": "tool", "content": "output1"},
                {
                    "role": "assistant",
                    "content": "Step 2",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "cat file"}'}}],
                },
                {"role": "tool", "content": "output2"},
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert len(result.steps) == 3
        # Find assistant steps
        assistant_steps = [s for s in result.steps if s.role == "assistant"]
        assert len(assistant_steps) == 2
        assert assistant_steps[0].tool_call == ("ls",)
        assert assistant_steps[1].tool_call == ("cat file",)

    def test_parse_trajectory_extracts_reasoning(self) -> None:
        """Test that reasoning_content is correctly extracted."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Check file",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
                    "reasoning_content": "I need to list files to understand the project structure",
                },
                {"role": "tool", "content": "output1"},
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert len(result.steps) == 2
        assistant_step = result.steps[1]
        assert assistant_step.role == "assistant"
        assert assistant_step.reasoning == "I need to list files to understand the project structure"

    def test_parse_trajectory_extracts_context_focus_questions_in_order(self) -> None:
        """Focus questions should stay aligned with multiple bash tool calls."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Check files",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": (
                                    '{"command": "ls", "context_focus_question": '
                                    '"Inspect the repository root."}'
                                ),
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "cat file"}',
                            },
                        },
                    ],
                },
                {"role": "tool", "content": "output1"},
            ],
        }

        result = parse_trajectory(trajectory_json)

        assistant_step = result.steps[1]
        assert assistant_step.role == "assistant"
        assert assistant_step.tool_call == ("ls", "cat file")
        assert assistant_step.context_focus_question == ("Inspect the repository root.", None)

    def test_parse_trajectory_missing_reasoning(self) -> None:
        """Test that missing reasoning_content is handled gracefully."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Check file",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
                },
                {"role": "tool", "content": "output1"},
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert len(result.steps) == 2
        assistant_step = result.steps[1]
        assert assistant_step.role == "assistant"
        assert assistant_step.reasoning is None

    def test_parse_trajectory_skips_format_errors(self) -> None:
        """Test that FormatError messages are skipped."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Invalid format",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": "invalid"}}],
                },
                {
                    "role": "user",
                    "content": "Format error message",
                    "extra": {"interrupt_type": "FormatError"},
                },
                {
                    "role": "assistant",
                    "content": "Correct format",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
                },
                {"role": "tool", "content": "output"},
            ],
        }

        result = parse_trajectory(trajectory_json)

        # FormatError message should be skipped
        user_steps = [s for s in result.steps if s.role == "user"]
        assert len(user_steps) == 1  # Only the first user message

    def test_parse_trajectory_multiple_tool_outputs_in_sequence(self) -> None:
        """Test parsing multiple tool outputs for a single assistant message."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Multiple commands",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                        {"function": {"name": "bash", "arguments": '{"command": "pwd"}'}},
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"returncode": 0, "output": "file1.py"}',
                },
                {
                    "role": "tool",
                    "content": '{"returncode": 0, "output": "/home/user"}',
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assistant_step = [s for s in result.steps if s.role == "assistant"][0]
        assert len(assistant_step.tool_output) == 2
        assert assistant_step.tool_output[0].output == "file1.py"
        assert assistant_step.tool_output[1].output == "/home/user"

    def test_parse_trajectory_tool_output_with_exception(self) -> None:
        """Test parsing tool output with exception info."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Test"},
                {
                    "role": "assistant",
                    "content": "Check file",
                    "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": '{"command": "cat missing"}'}}],
                },
                {
                    "role": "tool",
                    "content": '{"returncode": 1, "output": "", "exception_info": "FileNotFoundError"}',
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assistant_step = [s for s in result.steps if s.role == "assistant"][0]
        assert assistant_step.tool_output[0].returncode == 1
        assert assistant_step.tool_output[0].exception == "FileNotFoundError"

    def test_parse_trajectory_pr_description_extraction(self) -> None:
        """Test extraction of issue from PR description tags."""
        trajectory_json = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {
                    "role": "user",
                    "content": "Consider the following issue:\n<pr_description>This is the actual bug description\nwith multiple lines</pr_description>",
                },
            ],
        }

        result = parse_trajectory(trajectory_json)

        assert "actual bug description" in result.issue_description
