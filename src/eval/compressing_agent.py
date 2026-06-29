"""Compressing agent that applies compression during agent execution."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from minisweagent.agents.default import DefaultAgent

from src.agent.trajectory_parser import effective_assistant_text
from src.compression import (
    CompressionFilter,
    build_runtime_compression_context,
    has_context_focus_question,
    interpret_compression_response,
)
from src.eval.baselines import (
    AgentDietCoACTCompression,
    AgentDietCompression,
    CompressionAttempt,
    CompressionStrategy,
    LLMLingua2Compression,
    LongCodeZipCompression,
    SlidingWindow,
    SlidingWindowCoACTCompression,
    SWEPrunerCompression,
    VanillaCompression,
)

logger = logging.getLogger(__name__)

COMPRESSION_STATS_DEFAULTS: dict[str, Any] = {
    "total_time_ms": 0.0,
    "compressed_steps": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "cache_creation_input_tokens": 0,
    "total_tokens": 0,
    "input_cost_usd": 0.0,
    "cache_read_cost_usd": 0.0,
    "cache_creation_cost_usd": 0.0,
    "output_cost_usd": 0.0,
    "total_cost_usd": 0.0,
}


def _tool_step_id(messages: Sequence[Mapping[str, Any]], assistant_index: int) -> int:
    """Return the merged-step id for an assistant tool-call message index."""
    step_id = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if not message.get("tool_calls"):
            continue
        if index == assistant_index:
            return step_id
        step_id += 1
    return step_id


class CompressingAgent(DefaultAgent):
    """Agent that compresses tool outputs during execution.

    This agent overrides execute_actions to compress tool outputs
    in real-time, before they are added to the message history.
    """

    def __init__(
        self,
        strategy: CompressionStrategy,
        compression_stats: dict[str, Any],
        compression_filter: CompressionFilter | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize compressing agent.

        Args:
            strategy: The compression strategy to use.
            compression_stats: Dict to track compression statistics (modified in-place).
            compression_filter: Optional CompressionFilter to bypass compression for some outputs.
            *args: Passed to parent.
            **kwargs: Passed to parent.
        """
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.compression_filter = compression_filter
        self._compression_stats = compression_stats

    def _record_compression_attempt(self, attempt: CompressionAttempt | None) -> None:
        """Accumulate one already-priced compression call into compression_stats.

        Strategies hand back a :class:`CompressionAttempt` they priced
        themselves. ``None`` means the strategy short-circuited (no focus
        question, no-op strategy, length-gate hit) — nothing to record.
        Fully-zero attempts are also skipped so empty events stay invisible.
        """
        if attempt is None:
            return

        cost = attempt.cost
        usage = attempt.usage
        cost_total = float(cost.get("total_cost_usd", 0.0) or 0.0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if attempt.time_ms <= 0.0 and cost_total <= 0.0 and total_tokens <= 0:
            return

        stats = self._compression_stats
        stats["total_time_ms"] += attempt.time_ms

        stats["input_cost_usd"] += float(cost.get("input_cost_usd", 0.0))
        stats["cache_read_cost_usd"] += float(cost.get("cache_read_cost_usd", 0.0))
        stats["cache_creation_cost_usd"] += float(cost.get("cache_creation_cost_usd", 0.0))
        stats["output_cost_usd"] += float(cost.get("output_cost_usd", 0.0))
        stats["total_cost_usd"] += float(cost.get("total_cost_usd", 0.0))

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["cached_tokens"] += int(usage.get("cached_tokens", 0) or 0)
        stats["cache_creation_input_tokens"] += int(
            usage.get("cache_creation_input_tokens", 0) or 0
        )
        stats["total_tokens"] += prompt_tokens + completion_tokens

    def _record_agentdiet_audit(
        self,
        *,
        assistant_index: int,
        tool_index: int,
        original_assistant_text: str,
        reduced_assistant_text: str,
        original_tool_content: str,
        reduced_tool_content: str,
        time_ms: float,
    ) -> None:
        """Append one lightweight AgentDiet audit record to compression stats."""
        audit_events = self._compression_stats.setdefault("agentdiet_audit", [])
        if not isinstance(audit_events, list):
            return

        step_id = _tool_step_id(self.messages, assistant_index)
        audit_events.append(
            {
                "step_id": step_id,
                "assistant_index": assistant_index,
                "tool_index": tool_index,
                "assistant_before_chars": len(original_assistant_text),
                "assistant_after_chars": len(reduced_assistant_text),
                "tool_before_chars": len(original_tool_content),
                "tool_after_chars": len(reduced_tool_content),
                "tool_after_empty": not reduced_tool_content.strip(),
                "reduction_time_ms": time_ms,
            }
        )

    def _apply_agentdiet_reduction(self, reduction: Any) -> bool:
        if (
            reduction.assistant_index is None
            or reduction.tool_index is None
            or reduction.reduced_assistant_content is None
            or reduction.reduced_tool_content is None
            or not 0 <= reduction.assistant_index < len(self.messages)
            or not 0 <= reduction.tool_index < len(self.messages)
            or self.messages[reduction.assistant_index].get("role") != "assistant"
            or self.messages[reduction.tool_index].get("role") != "tool"
        ):
            return False

        original_assistant_msg = self.messages[reduction.assistant_index]
        original_assistant_text = effective_assistant_text(
            original_assistant_msg.get("content"),
            original_assistant_msg.get("reasoning_content"),
        )
        original_tool_content = self.messages[reduction.tool_index].get("content") or ""

        assistant_msg = {
            **self.messages[reduction.assistant_index],
            "content": reduction.reduced_assistant_content,
        }
        assistant_msg["extra"] = {
            **(assistant_msg.get("extra") or {}),
            "agentdiet_reduced": True,
        }
        agent_erased = getattr(reduction, "agent_erased", None)
        if isinstance(agent_erased, str) and agent_erased.strip():
            assistant_msg["agent_erased"] = agent_erased

        tool_msg = {
            **self.messages[reduction.tool_index],
            "content": reduction.reduced_tool_content,
        }
        original_tool_extra = self.messages[reduction.tool_index].get("extra") or {}
        tool_msg["extra"] = {
            **(tool_msg.get("extra") or {}),
            "raw_output": reduction.reduced_tool_content,
            "original_raw_output": (
                original_tool_extra.get("original_raw_output")
                if isinstance(original_tool_extra, Mapping)
                and original_tool_extra.get("original_raw_output") is not None
                else original_tool_extra.get("raw_output", original_tool_content)
            )
            or "",
            "agentdiet_reduced": True,
        }

        self.messages[reduction.assistant_index] = assistant_msg
        self.messages[reduction.tool_index] = tool_msg
        self._compression_stats["compressed_steps"] += 1
        reduced_assistant_text = effective_assistant_text(
            reduction.reduced_assistant_content,
            None,
        )
        self._record_agentdiet_audit(
            assistant_index=reduction.assistant_index,
            tool_index=reduction.tool_index,
            original_assistant_text=original_assistant_text,
            reduced_assistant_text=reduced_assistant_text,
            original_tool_content=original_tool_content,
            reduced_tool_content=reduction.reduced_tool_content,
            time_ms=reduction.attempt.time_ms,
        )
        return True

    def _apply_sliding_window_redaction(self, sliding_window: SlidingWindow) -> bool:
        """Redact historical tool messages selected by a sliding window.

        Args:
            sliding_window: Sliding-window component that selects old tool
                messages for redaction.

        Returns:
            True when at least one message was redacted.
        """
        indices_to_remove = sliding_window.get_message_indices_to_remove(self.messages)
        redacted_this_step = False
        for msg_idx in indices_to_remove:
            new_msg = {
                **self.messages[msg_idx],
                "content": sliding_window.removed_content,
            }
            new_msg["extra"] = {
                **new_msg["extra"],
                "raw_output": sliding_window.removed_content,
            }
            self.messages[msg_idx] = new_msg
            redacted_this_step = True
        return redacted_this_step

    def _run_agentdiet_reduction(
        self,
        strategy: AgentDietCompression | AgentDietCoACTCompression,
    ) -> None:
        """Run AgentDiet history reduction and record its cost."""
        reduction = strategy.reduce_message_in_history(self.messages)
        self._record_compression_attempt(reduction.attempt)
        self._apply_agentdiet_reduction(reduction)

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions with compression applied to outputs.

        Args:
            message: The message containing actions to execute.

        Returns:
            List of observation messages.
        """
        actions = message.get("extra", {}).get("actions", [])

        if isinstance(self.strategy, VanillaCompression):
            return super().execute_actions(message)

        elif isinstance(self.strategy, SlidingWindow):
            result = super().execute_actions(message)

            if self._apply_sliding_window_redaction(self.strategy):
                self._compression_stats["compressed_steps"] += 1

            return result

        elif isinstance(self.strategy, SlidingWindowCoACTCompression):
            pre_compressed_steps = int(self._compression_stats["compressed_steps"])
            outputs = [self.env.execute(action) for action in actions]
            original_outputs = [str(o.get("output", "")) for o in outputs]

            compressed_outputs = self._compress_outputs(outputs, message)

            compressed_messages = self.model.format_observation_messages(
                message, compressed_outputs, self.get_template_vars()
            )
            result = self.add_messages(*compressed_messages)
            self._stash_original_tool_outputs(original_outputs)

            redacted_this_step = self._apply_sliding_window_redaction(self.strategy.sliding_window)
            if (
                redacted_this_step
                and int(self._compression_stats["compressed_steps"]) == pre_compressed_steps
            ):
                self._compression_stats["compressed_steps"] += 1

            return result

        elif isinstance(self.strategy, AgentDietCoACTCompression):
            outputs = [self.env.execute(action) for action in actions]
            original_outputs = [str(o.get("output", "")) for o in outputs]

            compressed_outputs = self._compress_outputs(outputs, message)

            compressed_messages = self.model.format_observation_messages(
                message, compressed_outputs, self.get_template_vars()
            )
            result = self.add_messages(*compressed_messages)
            self._stash_original_tool_outputs(original_outputs)

            self._run_agentdiet_reduction(self.strategy)

            return result

        elif isinstance(self.strategy, AgentDietCompression):
            result = super().execute_actions(message)

            self._run_agentdiet_reduction(self.strategy)

            return result

        elif isinstance(
            self.strategy,
            (SWEPrunerCompression, LLMLingua2Compression, LongCodeZipCompression),
        ):
            # SWEPruner, LLMLingua-2 and LongCodeZip return final compressed text
            # directly; skip the interpret_compression_response envelope used by CoACT.
            outputs = [self.env.execute(action) for action in actions]
            original_outputs = [str(o.get("output", "")) for o in outputs]

            compressed_outputs = self._compress_outputs(
                outputs,
                message,
                resolve_response=False,
            )

            compressed_messages = self.model.format_observation_messages(
                message, compressed_outputs, self.get_template_vars()
            )
            result = self.add_messages(*compressed_messages)
            self._stash_original_tool_outputs(original_outputs)
            return result

        else:
            # Compressible model-backed strategy: compress each output.
            outputs = [self.env.execute(action) for action in actions]
            original_outputs = [str(o.get("output", "")) for o in outputs]

            # Compress outputs synchronously
            compressed_outputs = self._compress_outputs(outputs, message)

            compressed_messages = self.model.format_observation_messages(
                message, compressed_outputs, self.get_template_vars()
            )
            result = self.add_messages(*compressed_messages)
            self._stash_original_tool_outputs(original_outputs)
            return result

    def _stash_original_tool_outputs(self, original_outputs: list[str]) -> None:
        """Save raw tool outputs onto the just-added tool messages.

        The compressor replaces a tool message's ``content`` with the compressed
        string before it is added to ``self.messages`` — the original output is
        otherwise lost from the saved trajectory. This stashes it under
        ``extra.original_tool_output`` for downstream DAGGER training-data
        construction (see ``scripts/prepare_dagger_data.py``).

        Walks the tail of ``self.messages`` for the most recently added ``tool``
        messages and assigns ``extra.original_tool_output`` in order.
        """
        if not original_outputs:
            return
        n = len(original_outputs)
        tool_indices: list[int] = []
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "tool":
                tool_indices.append(i)
                if len(tool_indices) == n:
                    break
            else:
                if tool_indices:
                    break
        tool_indices.reverse()
        if len(tool_indices) != n:
            logger.warning(
                f"_stash_original_tool_outputs: expected {n} trailing tool messages, "
                f"found {len(tool_indices)}; original_tool_output not stashed"
            )
            return
        for tool_idx, original in zip(tool_indices, original_outputs, strict=True):
            target = self.messages[tool_idx]
            extra = dict(target.get("extra") or {})
            extra["original_tool_output"] = original
            self.messages[tool_idx] = {**target, "extra": extra}

    def _compress_outputs(
        self,
        outputs: list[dict],
        assistant_message: dict,
        resolve_response: bool = True,
    ) -> list[dict]:
        """Compress tool outputs using the strategy.

        Args:
            outputs: List of output dicts from env.execute().
            assistant_message: The assistant message that triggered these outputs.
            resolve_response: When True (default, used by CoACT), pass
                the strategy's response through ``interpret_compression_response``
                to resolve ``<keep_all/>`` / block-id envelopes. SWEPruner
                returns final pruned text directly and sets this to False.

        Returns:
            List of compressed output dicts.
        """
        from src.agent.miniswe_runner import MiniSWERunner

        compressed_outputs = []
        compressed_this_step = False

        for output in outputs:
            original_content = output.get("output", "")
            context = build_runtime_compression_context(
                self.messages,
                assistant_message,
                output,
            )
            strategy_invoked = False

            attempt: CompressionAttempt | None = None
            if self.strategy.consumes_cfq and not has_context_focus_question(
                context["context_focus_question"]
            ):
                compressed_content = original_content
            elif self.compression_filter is not None:
                filter_reason = self.compression_filter.get_filter_reason(
                    tool_calls=context["tool_call"],
                    tool_output_str=context["tool_output_for_prompt"],
                    step_index=len(self.messages),
                )
                if filter_reason is not None:
                    compressed_content = original_content
                else:
                    compressed_content, attempt = self.strategy.compress(
                        original_content,
                        context,
                    )
                    strategy_invoked = True
            else:
                compressed_content, attempt = self.strategy.compress(
                    original_content,
                    context,
                )
                strategy_invoked = True

            self._record_compression_attempt(attempt)
            if strategy_invoked and resolve_response:
                resolution = interpret_compression_response(
                    compressed_content,
                    original_output=original_content,
                    numbered_output=context["tool_output_for_prompt"],
                )
                compressed_content = resolution.effective_output

            if resolve_response:
                compressed_content = MiniSWERunner.annotate_compressed_output(
                    compressed_output=compressed_content,
                    original_output=original_content,
                )

            if compressed_content != original_content:
                compressed_this_step = True

            compressed_output = output.copy()
            compressed_output["output"] = compressed_content
            compressed_outputs.append(compressed_output)

        if compressed_this_step:
            self._compression_stats["compressed_steps"] += 1

        return compressed_outputs
