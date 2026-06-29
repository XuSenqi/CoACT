"""Tests for the repo-local LiteLLM model with context-focus tool support."""

from types import SimpleNamespace
from unittest.mock import patch

from src.agent.context_focus_litellm_model import (
    CONTEXT_FOCUS_BASH_TOOL,
    ContextFocusLitellmModel,
    parse_toolcall_actions_with_context_focus,
)


def test_query_uses_context_focus_bash_tool_schema() -> None:
    """The local LiteLLM wrapper should advertise the extended bash schema."""
    model = ContextFocusLitellmModel(model_name="openai/test-model")

    with patch("src.agent.context_focus_litellm_model.litellm.completion") as mock_completion:
        model._query([{"role": "user", "content": "hello"}])

    assert mock_completion.call_args.kwargs["tools"] == [CONTEXT_FOCUS_BASH_TOOL]
    parameters = mock_completion.call_args.kwargs["tools"][0]["function"]["parameters"]
    assert "context_focus_question" in parameters["properties"]
    assert parameters["required"] == ["command"]
    description = parameters["properties"]["context_focus_question"]["description"]
    assert "Describe the specific information you need" in description
    assert "compressed to keep only the relevant parts" in description
    assert "Be precise" in description
    assert "validate() method signature" in description
    assert "Omit this field to receive the full uncompressed output" in description
    assert "previous compressed result was missing details you need" in description


def test_parse_actions_preserves_context_focus_question() -> None:
    """Action parsing should preserve optional focus-question metadata."""
    tool_calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="bash",
                arguments=(
                    '{"command":"cat file.py","context_focus_question":"Inspect the login path."}'
                ),
            ),
        ),
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="bash", arguments='{"command":"grep TODO file.py"}'),
        ),
    ]

    actions = parse_toolcall_actions_with_context_focus(
        tool_calls,
        format_error_template="{{ error }}",
    )

    assert actions == [
        {
            "command": "cat file.py",
            "context_focus_question": "Inspect the login path.",
            "tool_call_id": "call-1",
        },
        {
            "command": "grep TODO file.py",
            "context_focus_question": None,
            "tool_call_id": "call-2",
        },
    ]
