"""LiteLLM model wrapper with context-focus support for bash tool calls."""

import json
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel

CONTEXT_FOCUS_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "context_focus_question": {
                    "type": "string",
                    "description": (
                        "Describe the specific information you need from this command's "
                        "output. The output will be compressed to keep only the relevant "
                        "parts. Be precise: e.g. 'find the validate() method signature "
                        "and what exception it raises' instead of 'read the file'. "
                        "Omit this field to receive the full uncompressed output when a "
                        "previous compressed result was missing details you need."
                    ),
                },
            },
            "required": ["command"],
        },
    },
}


def _build_format_error(
    format_error_template: str,
    *,
    error: str,
) -> FormatError:
    """Build a Mini-SWE-Agent format error message.

    Args:
        format_error_template: Template used to render the error message.
        error: Human-readable format error text.

    Returns:
        `FormatError` carrying the rendered user interrupt message.
    """
    return FormatError(
        {
            "role": "user",
            "content": Template(
                format_error_template,
                undefined=StrictUndefined,
            ).render(
                actions=[],
                error=error,
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
    )


def _normalize_context_focus_question(raw_focus_question: Any) -> tuple[str | None, str]:
    """Normalize one optional `context_focus_question` value.

    Args:
        raw_focus_question: Parsed JSON value for `context_focus_question`.

    Returns:
        Tuple of `(normalized_question, error_message)`. The question is
        stripped and empty strings become `None`. The error message is empty
        when the value is valid.
    """
    if raw_focus_question is None:
        return None, ""
    if not isinstance(raw_focus_question, str):
        return (
            None,
            "Optional 'context_focus_question' argument in bash tool call must be a string or null.",
        )

    normalized_focus_question = raw_focus_question.strip()
    return normalized_focus_question or None, ""


def parse_toolcall_actions_with_context_focus(
    tool_calls: list[Any],
    *,
    format_error_template: str,
) -> list[dict[str, Any]]:
    """Parse bash tool calls while preserving `context_focus_question`.

    Args:
        tool_calls: Tool calls returned by the model response.
        format_error_template: Template used to construct format-error messages.

    Returns:
        Parsed action dictionaries with `command`, `tool_call_id`, and optional
        `context_focus_question`.

    Raises:
        FormatError: If tool calls are missing or malformed.
    """
    if not tool_calls:
        raise _build_format_error(
            format_error_template,
            error=(
                "No tool calls found in the response. Every response MUST include "
                "at least one tool call."
            ),
        )

    actions: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        error_msg = ""
        args: Any = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as exc:
            error_msg = f"Error parsing tool call arguments: {exc}."

        if tool_call.function.name != "bash":
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."

        context_focus_question = None
        if isinstance(args, dict) and "context_focus_question" in args:
            context_focus_question, focus_error = _normalize_context_focus_question(
                args["context_focus_question"]
            )
            error_msg += focus_error

        if error_msg:
            raise _build_format_error(
                format_error_template,
                error=error_msg.strip(),
            )

        actions.append(
            {
                "command": args["command"],
                "context_focus_question": context_focus_question,
                "tool_call_id": tool_call.id,
            }
        )
    return actions


class ContextFocusLitellmModel(LitellmModel):
    """LiteLLM model wrapper that extends bash tool calls with focus metadata."""

    def _query(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Query LiteLLM with the extended bash tool schema.

        Args:
            messages: Prepared API messages.
            **kwargs: Extra LiteLLM completion parameters.

        Returns:
            LiteLLM completion response object.
        """
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[CONTEXT_FOCUS_BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as exc:
            exc.message += (
                " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            )
            raise exc

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        """Parse tool calls while preserving `context_focus_question`.

        Args:
            response: LiteLLM completion response.

        Returns:
            Parsed bash actions with optional focus metadata.
        """
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions_with_context_focus(
            tool_calls,
            format_error_template=self.config.format_error_template,
        )
