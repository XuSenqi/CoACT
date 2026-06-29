"""Prompt templates for CoACT.

Provides all prompt templates used by the system:
- Observation template (without length filtering)
- Compression prompt for the SLM
- Agent context-focus guidance
"""

from jinja2 import StrictUndefined, Template

# Observation Template (without length filtering)
# Default mini.yaml truncates output > 10000 chars, this template preserves full output
OBSERVATION_TEMPLATE = """{
  "returncode": {{ output.returncode }},
  "output": {{ output.output | tojson }}
  {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
}"""

# Compression Prompt Template (uses Jinja2 syntax)
COMPRESSION_PROMPT = """You are a context compression assistant. Compress the tool output while preserving the evidence the agent needs for the correct next tool call.

## Global Task Goal
{{ goal or "(not provided)" }}

## Context Focus Question
{{ context_focus_question }}

## Tool Call Executed
{{ tool_call }}

## Original Tool Output
{{ tool_output or "(empty)" }}

## Instructions
Return EXACTLY one JSON object with fields in this order:
1. `"type"`
2. `"content"`

Global rules:
- The compressed output must be less than or equal to the original length.
- Return JSON only. Do NOT add commentary, headings, markdown fences, or wrapper text.
- Prioritize information preservation over aggressive compression. When in doubt, keep the content.

Choose exactly one case:

Case 1: No compression needed
- Use when the output is already concise or when compression may remove needed evidence.
- Output Example: `{"type":"unchanged","content":null}`
- `"content"` must be `null`.

Case 2: Plain-text compression
- Use for summaries of non-code outputs such as listings, passing test summaries, stats, and logs where exact lines are not needed.
- Output Example: `{"type":"plain","content":"All selected tests passed; no failure traceback was present."}`
- `"content"` must contain only the compressed tool output content.
- Do not include line numbers like `1>` or markdown code fences.

Case 3: Code-snippet compression
- Use for source code, tests, diffs/patches, grep/rg context, stack traces, or any output where line-level structure, syntax, or adjacency may matter.
- Output Example: `{"type":"code","content":["1-5:license header and imports","40-58:unrelated helper function"]}`
- `"content"` must be a non-empty JSON array of sorted, non-overlapping omit ranges within Original Tool Output.
- Entry format: `"N:summary"` or `"N-M:summary"`. Lines not covered by omit ranges are kept verbatim for the agent.
- Omit only lines clearly irrelevant to the CFQ, do not omit all nonblank original lines. The agent must see verbatim evidence, not only `(compressed N lines: ...)` markers.
- If the CFQ asks to show/examine a class/function/method including X, do not omit that class/function/method body.
- For broad reads like "read the file", "understand the implementation", or "understand expected behavior", compress conservatively and keep potentially relevant code/test bodies.
- Keep enclosing function, class, and control-flow structure needed to understand any kept line.
- If relevance is unclear, use `"unchanged"`.

## Compressed Output
"""

# Agent context-focus guidance injected into the instance template
CONTEXT_FOCUS_GUIDANCE = (
    "3. To save context, always provide a `context_focus_question` string in your bash tool "
    "calls. State the specific fact, value, or structure you need - not the general intent. "
    "For example, 'find the method signature of validate() and what exception it raises' "
    "is effective; 'read the file' or 'understand the code' is too vague and will cause "
    "important details to be lost. The tool output will be compressed to only keep the "
    "parts relevant to your question. "
    "If a previous compressed result was missing details you need, omit "
    "`context_focus_question` to receive the full uncompressed output."
)

TRAJECTORY_COLLECTION_NOTE = (
    "4. During trajectory collection, the bash tool's `context_focus_question` is recorded "
    "for training only. It will not trigger compression yet, so tool outputs will still be "
    "returned in full."
)

# Transparent-compression notice for task-agnostic compressors (e.g. LLMLingua-2).
# Unlike CONTEXT_FOCUS_GUIDANCE this does NOT mention `context_focus_question`, because the
# agent has no control over the compression: outputs are compressed unconditionally.
COMPRESSION_NOTICE_GUIDANCE = (
    "3. To save context, your tool outputs are automatically compressed before you see them: "
    "less salient words are dropped, so observations may read as condensed or slightly "
    "ungrammatical telegraphic text. Treat each observation as a faithful but lossy summary of "
    "the real command output, and reason from its overall content rather than its exact wording "
    "or formatting."
)

# AgentDiet Reduction Prompt Template (uses Jinja2 syntax)
AGENTDIET_REDUCTION_PROMPT = """You will analyze and compress a given step in a trajectory of an AI agent solving a software bug.

In the trajectory, each step is marked in <step id="..."></step>.
The agent will think in <think>, call external tools as marked in <call tool="..."></call>. Its result is marked in <result></result> within the <step> tag.

Your job is to compress the text within the TARGET step (id="{{ target_step_id }}") to avoid harming efficiency, typically shortening it to 20%-50% of the original length.
Meanwhile, keep the compressed text useful such that you are able to continue the trajectory as close as the original path.

- You should ONLY remove redundant texts, which are either irrelevant to future steps or duplicated by other texts in the trajectory.
- Replace the text to remove to "..." and a short takeaway, e.g. "... (same as the content below)".
- You should keep the original structure unchanged, e.g., XML tags, Python indentation, JSON formats and line numbers.
- Again, keep useful details in the original content unchanged, e.g., XML tags, Python indentation and line numbers.

Typical examples:
- If the step opens a huge file but only one part is necessary for future steps, replace other parts to "... (unrelated function XXX, YYY)".
- If the step runs a verbose test script and everything goes fine, replace the verbose part to "... (expected output)".
- If the step uses str_replace_editor to modify a file and the content can be inferred by the content after it, replace the tool call argument to "... (see results below)".

Task goal: {{ goal }}

Trajectory window:
{% for step_id, think_text, call_text, result_text in context_steps -%}
<step id="{{ step_id }}"{% if step_id == target_step_id %} [TARGET]{% endif %}>
<think>
{{ think_text }}
</think>
{{ call_text }}
<result>
{{ result_text }}
</result>
</step>
{% if not loop.last %}
{% endif -%}
{% endfor %}
You should only process the text within the <step> tag with the given id.
Please output the completely rewritten <step> block for the TARGET step. STOP OUTPUT IMMEDIATELY AFTER </step>."""


_COMPRESSION_PROMPT_TEMPLATE = Template(
    COMPRESSION_PROMPT,
    undefined=StrictUndefined,
)

_AGENTDIET_REDUCTION_PROMPT_TEMPLATE = Template(
    AGENTDIET_REDUCTION_PROMPT,
    undefined=StrictUndefined,
)


def render_agentdiet_reduction_prompt(
    goal: str,
    context_steps: list[tuple[int, str, str, str]],
    target_step_id: int,
) -> str:
    """Render the AgentDiet reflection prompt for one reduction attempt.

    Args:
        goal: The global task goal.
        context_steps: List of ``(step_id, think_text, call_text, result_text)`` tuples
            forming the trajectory window shown to the reflector.
        target_step_id: Step id within ``context_steps`` to mark as TARGET and rewrite.

    Returns:
        The rendered reflection prompt.
    """
    return _AGENTDIET_REDUCTION_PROMPT_TEMPLATE.render(
        goal=goal,
        context_steps=context_steps,
        target_step_id=target_step_id,
    )


def render_compression_prompt(
    goal: str,
    context_focus_question: tuple[str | None, ...] | str | None,
    tool_output: str,
    tool_call: tuple[str, ...] | None = None,
) -> str:
    """Render the compression prompt with provided values.

    Args:
        goal: The global task goal
        context_focus_question: The focus question(s) attached to the tool call
        tool_output: The original tool output to compress
        tool_call: The tool call(s) that produced this output

    Returns:
        The rendered compression prompt
    """
    return _COMPRESSION_PROMPT_TEMPLATE.render(
        goal=goal,
        context_focus_question=_format_context_focus_question(context_focus_question),
        tool_output=tool_output,
        tool_call=_format_tool_call(tool_call),
    )


def _format_context_focus_question(
    context_focus_question: tuple[str | None, ...] | str | None,
) -> str:
    """Render context-focus question(s) for the compression prompt.

    Args:
        context_focus_question: One focus question or a tuple aligned with tool calls.

    Returns:
        Prompt-ready string representation.
    """
    if isinstance(context_focus_question, str):
        normalized_question = context_focus_question.strip()
        return normalized_question or "(none)"

    if context_focus_question is None:
        return "(none)"

    normalized_questions = [
        question.strip()
        for question in context_focus_question
        if isinstance(question, str) and question.strip()
    ]
    if not normalized_questions:
        return "(none)"
    if len(normalized_questions) == 1:
        return normalized_questions[0]
    return "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(normalized_questions, start=1)
    )


def _format_tool_call(tool_call: tuple[str, ...] | None) -> str:
    """Render tool call commands for the compression prompt.

    Args:
        tool_call: Tool call command tuple.

    Returns:
        Prompt-ready tool call representation.
    """
    if not tool_call:
        return "(none)"
    return "\n".join(f"$ {command}" for command in tool_call)
