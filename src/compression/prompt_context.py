"""Shared prompt-context helpers for compression training and inference."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.agent.trajectory_parser import ToolOutput, TrajectoryStep

_PR_DESCRIPTION_RE = re.compile(r"<pr_description>(.*?)</pr_description>", re.DOTALL)
_PR_PREFIX = "Consider the following PR description:"


CompressionPromptContext = dict[str, Any]


def _normalize_context_focus_question(value: object) -> str | None:
    """Normalize an optional context-focus question value.

    Args:
        value: Raw value to normalize.

    Returns:
        Stripped focus question, or `None` if missing/blank/non-string.
    """
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    return normalized_value or None


def has_context_focus_question(
    context_focus_question: Sequence[str | None] | str | None,
) -> bool:
    """Return whether a context-focus question is available.

    Args:
        context_focus_question: One focus question or a sequence of them.

    Returns:
        `True` if at least one non-empty focus question is present.
    """
    if isinstance(context_focus_question, str):
        return bool(context_focus_question.strip())
    if context_focus_question is None:
        return False
    return any(question is not None and question.strip() for question in context_focus_question)


def number_prompt_lines(text: str) -> str:
    """Add 1-based line numbers to prompt-visible text.

    Args:
        text: Unnumbered text.

    Returns:
        Text where each line is prefixed with ``N> ``.
    """
    return "\n".join(
        f"{line_number}> {line}" for line_number, line in enumerate(text.split("\n"), start=1)
    )


def extract_goal_from_text(content: str) -> str:
    """Extract the cleaned task goal from a user message."""
    match = _PR_DESCRIPTION_RE.search(content)
    if not match:
        return content

    goal = match.group(1).strip()
    if _PR_PREFIX in goal:
        goal = goal.split(_PR_PREFIX, 1)[-1].strip()
    return goal


def extract_goal_from_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Extract the first user goal from a message history."""
    for msg in messages:
        if msg.get("role") == "user":
            return extract_goal_from_text(str(msg.get("content", "")))
    return ""


def format_tool_outputs(tool_outputs: Sequence[ToolOutput]) -> str:
    """Format all tool outputs from one trajectory step."""
    if len(tool_outputs) == 1:
        unnumbered_outputs = tool_outputs[0].output or ""
    else:
        unnumbered_outputs = "\n".join(
            f"=== Output {i} ===\n{output.output or ''}"
            for i, output in enumerate(tool_outputs, start=1)
        )
    return number_prompt_lines(unnumbered_outputs)


def build_step_compression_context(step: TrajectoryStep) -> CompressionPromptContext:
    """Build the shared compression prompt context for one training step."""
    return {
        "goal": step.goal,
        "context_focus_question": step.context_focus_question,
        "tool_call": step.tool_call,
        "tool_output_for_prompt": format_tool_outputs(step.tool_output),
    }


def build_runtime_compression_context(
    messages: Sequence[Mapping[str, Any]],
    assistant_message: Mapping[str, Any],
    output: Mapping[str, Any],
) -> CompressionPromptContext:
    """Build the shared compression prompt context during live evaluation."""
    actions = assistant_message.get("extra", {}).get("actions") or []
    tool_call = tuple(action.get("command", "") for action in actions) or ()
    context_focus_question = tuple(
        _normalize_context_focus_question(action.get("context_focus_question"))
        for action in actions
    )

    return {
        "goal": extract_goal_from_messages(messages),
        "context_focus_question": context_focus_question,
        "tool_call": tool_call,
        "tool_output_for_prompt": number_prompt_lines(str(output.get("output", ""))),
    }
