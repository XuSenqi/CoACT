"""Tests for compression behavior inside CompressingAgent."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from minisweagent.agents.default import DefaultAgent

from src.eval.baselines import (
    AgentDietCoACTCompression,
    AgentDietCompression,
    AgentDietReductionResult,
    CompressionAttempt,
    SlidingWindow,
    SlidingWindowCoACTCompression,
    SWEPrunerCompression,
)
from src.eval.compressing_agent import COMPRESSION_STATS_DEFAULTS, CompressingAgent


def _agentdiet_history() -> list[dict]:
    return [
        {"role": "user", "content": "Fix bug"},
        {
            "role": "assistant",
            "content": "Inspect old file",
            "reasoning_content": "Need the old implementation details.",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "cat old.py"}),
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps({"returncode": 0, "output": "old output"}),
            "extra": {"raw_output": "old output"},
        },
        {
            "role": "assistant",
            "content": "Inspect new file",
            "reasoning_content": "Need the latest context before editing.",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "cat new.py"}),
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps({"returncode": 0, "output": "new output"}),
            "extra": {"raw_output": "new output"},
        },
    ]


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    def model_dump(self) -> dict:
        return {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "cache_creation_input_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 4},
            }
        }


class TestCompressingAgent:
    """Tests for CompressingAgent internals."""

    def test_compressed_steps_counts_per_assistant_step(self) -> None:
        """Multiple compressed tool outputs in one turn count as one step."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.compress.side_effect = [
            ('{"type":"plain","content":"short-a"}', CompressionAttempt(time_ms=10.0)),
            ('{"type":"plain","content":"short-b"}', CompressionAttempt(time_ms=20.0)),
        ]
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [
            {"output": "very long output a"},
            {"output": "very long output b"},
        ]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {
                "actions": [
                    {
                        "command": "cat a.py",
                        "context_focus_question": "Inspect the relevant code in a.py.",
                    },
                    {
                        "command": "cat b.py",
                        "context_focus_question": "Inspect the relevant code in b.py.",
                    },
                ],
            },
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        # Output should have compression hint prepended
        assert compressed_outputs[0]["output"].endswith("short-a")
        assert compressed_outputs[1]["output"].endswith("short-b")
        assert "[Original tool output has been compressed]" in compressed_outputs[0]["output"]
        assert agent._compression_stats["total_time_ms"] == 30.0
        assert agent._compression_stats["compressed_steps"] == 1

    def test_unchanged_json_restores_original_output(self) -> None:
        """Unchanged JSON should pass the original tool output to the agent."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.compress.return_value = (
            '{"type":"unchanged","content":null}',
            CompressionAttempt(time_ms=12.0),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "exact original output"}]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {
                "actions": [
                    {
                        "command": "cat a.py",
                        "context_focus_question": "Inspect the relevant code in a.py.",
                    }
                ]
            },
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        assert compressed_outputs[0]["output"] == "exact original output"
        assert agent._compression_stats["total_time_ms"] == 12.0
        assert agent._compression_stats["compressed_steps"] == 0

    def test_valid_citations_reconstruct_original_lines(self) -> None:
        """Valid citations should reconstruct prompt-visible original lines."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.compress.return_value = (
            '{"type":"code","content":["2:beta line"]}',
            CompressionAttempt(time_ms=8.0),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "alpha\nbeta"}]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {
                "actions": [
                    {
                        "command": "cat a.py",
                        "context_focus_question": "Inspect the relevant code in a.py.",
                    }
                ]
            },
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        assert compressed_outputs[0]["output"] == "\n".join(
            [
                "[Original tool output has been compressed]",
                "alpha",
                "(compressed 1 lines: beta line)",
            ]
        )
        assert agent._compression_stats["compressed_steps"] == 1

    def test_non_adjacent_citations_remain_separate(self) -> None:
        """Non-adjacent omit ranges should remain separate without merging."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.compress.return_value = (
            '{"type":"code","content":["1:first line","3:third line"]}',
            CompressionAttempt(time_ms=8.0),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "alpha\nbeta\ngamma"}]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {
                "actions": [
                    {
                        "command": "cat a.py",
                        "context_focus_question": "Inspect the relevant code in a.py.",
                    }
                ]
            },
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        assert compressed_outputs[0]["output"] == "\n".join(
            [
                "[Original tool output has been compressed]",
                "(compressed 1 lines: first line)",
                "beta",
                "(compressed 1 lines: third line)",
            ]
        )
        assert agent._compression_stats["compressed_steps"] == 1

    def test_invalid_citations_fall_back_to_original_output(self) -> None:
        """Invalid citations should be treated like keep-original behavior."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.compress.return_value = (
            '{"type":"code","content":["2-"]}',
            CompressionAttempt(time_ms=8.0),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "alpha\nbeta"}]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {
                "actions": [
                    {
                        "command": "cat a.py",
                        "context_focus_question": "Inspect the relevant code in a.py.",
                    }
                ]
            },
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        assert compressed_outputs[0]["output"] == "alpha\nbeta"
        assert agent._compression_stats["compressed_steps"] == 0

    def test_missing_context_focus_question_bypasses_compression(self) -> None:
        """CFQ-consuming strategies skip compression when no focus question is present."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.consumes_cfq = True
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "exact original output"}]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {"actions": [{"command": "cat a.py"}]},
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        assert compressed_outputs[0]["output"] == "exact original output"
        agent.strategy.compress.assert_not_called()
        assert agent._compression_stats["total_time_ms"] == 0.0
        assert agent._compression_stats["compressed_steps"] == 0

    def test_task_agnostic_strategy_compresses_without_focus_question(self) -> None:
        """consumes_cfq=False strategies (e.g. LLMLingua-2) compress even with no CFQ."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.consumes_cfq = False
        agent.strategy.compress.return_value = (
            "compressed output",
            CompressionAttempt(time_ms=5.0),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "x" * 1024}]
        assistant_message = {
            "reasoning_content": "Inspect files.",
            # No context_focus_question: a task-agnostic compressor must still run.
            "extra": {"actions": [{"command": "cat a.py"}]},
        }

        compressed_outputs = agent._compress_outputs(
            outputs, assistant_message, resolve_response=False
        )

        agent.strategy.compress.assert_called_once()
        assert compressed_outputs[0]["output"] == "compressed output"
        assert agent._compression_stats["total_time_ms"] == 5.0
        assert agent._compression_stats["compressed_steps"] == 1

    def test_sliding_window_redaction_counts_as_compressed_step(self) -> None:
        """Sliding-window redaction should count as one compressed step."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = SlidingWindow(window_size=1)
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "tool",
                "content": "long tool output",
                "extra": {
                    "raw_output": "long tool output",
                    "returncode": 0,
                    "timestamp": 0.0,
                    "exception_info": None,
                },
            },
        ]
        agent.logger = MagicMock()
        agent.model = MagicMock()
        agent.env = MagicMock()
        agent.get_template_vars = lambda **kwargs: {}
        agent.env.execute.return_value = {"output": "new output"}
        agent.model.format_observation_messages.return_value = [
            {"role": "tool", "content": "new output"}
        ]
        message = {"extra": {"actions": [{"command": "cat file.py"}]}}

        result = agent.execute_actions(message)

        assert result == [{"role": "tool", "content": "new output"}]
        assert agent._compression_stats["compressed_steps"] == 1
        assert agent.messages[1]["content"] == SlidingWindow.removed_content

    def test_sliding_window_redaction_is_what_next_query_receives(self) -> None:
        """The next model query should only see the redacted tool content."""

        class ObservationModel:
            def format_observation_messages(self, message, outputs, template_vars):
                output = outputs[0]
                return [
                    {
                        "role": "tool",
                        "content": output["output"],
                        "extra": {
                            "raw_output": output["output"],
                            "returncode": output.get("returncode", 0),
                            "timestamp": 0.0,
                            "exception_info": output.get("exception_info"),
                        },
                    }
                ]

        class RecordingModel:
            def __init__(self) -> None:
                self.seen_messages = None

            def query(self, messages):
                self.seen_messages = messages
                return {"role": "assistant", "content": "", "extra": {"actions": []}}

        agent = object.__new__(CompressingAgent)
        agent.strategy = SlidingWindow(window_size=1)
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": "first"},
            {
                "role": "tool",
                "content": "old output",
                "extra": {
                    "raw_output": "old output",
                    "returncode": 0,
                    "timestamp": 0.0,
                    "exception_info": None,
                },
            },
        ]
        agent.model = ObservationModel()
        agent.env = SimpleNamespace(
            execute=lambda action: {
                "output": "new output",
                "returncode": 0,
                "exception_info": None,
            }
        )
        agent.get_template_vars = lambda **kwargs: {}
        agent.config = SimpleNamespace(step_limit=0, cost_limit=0, output_path=None)
        agent.n_calls = 0
        agent.cost = 0.0
        agent.logger = MagicMock()

        agent.execute_actions({"extra": {"actions": [{"command": "cat file.py"}]}})

        recording_model = RecordingModel()
        agent.model = recording_model
        DefaultAgent.query(agent)

        tool_messages = [msg for msg in recording_model.seen_messages if msg.get("role") == "tool"]

        assert tool_messages[0]["content"] == SlidingWindow.removed_content
        assert tool_messages[0]["extra"]["raw_output"] == SlidingWindow.removed_content
        assert tool_messages[1]["content"] == "new output"
        assert all("old output" not in msg["content"] for msg in tool_messages)

    def test_sliding_window_counts_tool_messages_not_assistant_turns(self) -> None:
        """Multiple tool calls in one turn still consume multiple window slots."""

        class ObservationModel:
            def format_observation_messages(self, message, outputs, template_vars):
                messages = []
                for output in outputs:
                    messages.append(
                        {
                            "role": "tool",
                            "content": output["output"],
                            "extra": {
                                "raw_output": output["output"],
                                "returncode": output.get("returncode", 0),
                                "timestamp": 0.0,
                                "exception_info": output.get("exception_info"),
                            },
                        }
                    )
                return messages

        outputs = iter(
            [
                {"output": "out-1", "returncode": 0, "exception_info": None},
                {"output": "out-2", "returncode": 0, "exception_info": None},
            ]
        )

        agent = object.__new__(CompressingAgent)
        agent.strategy = SlidingWindow(window_size=1)
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [{"role": "user", "content": "Fix bug"}]
        agent.model = ObservationModel()
        agent.env = SimpleNamespace(execute=lambda action: next(outputs))
        agent.get_template_vars = lambda **kwargs: {}
        agent.logger = MagicMock()

        agent.execute_actions(
            {
                "extra": {
                    "actions": [
                        {"command": "cmd-1"},
                        {"command": "cmd-2"},
                    ]
                }
            }
        )

        tool_messages = [msg for msg in agent.messages if msg.get("role") == "tool"]

        assert [msg["content"] for msg in tool_messages] == [SlidingWindow.removed_content, "out-2"]

    def test_sliding_window_coact_compresses_new_output_then_redacts_history(self) -> None:
        """RQ2 combo should run CoACT before the sliding-window history redaction."""

        class ObservationModel:
            def format_observation_messages(self, message, outputs, template_vars):
                output = outputs[0]
                return [
                    {
                        "role": "tool",
                        "content": output["output"],
                        "extra": {
                            "raw_output": output["output"],
                            "returncode": output.get("returncode", 0),
                            "timestamp": 0.0,
                            "exception_info": output.get("exception_info"),
                        },
                    }
                ]

        agent = object.__new__(CompressingAgent)
        observation_strategy = MagicMock()

        def compress(tool_output, context):
            assert tool_output == "new raw output"
            assert context["context_focus_question"] == ("Inspect new file.",)
            assert agent.messages[1]["content"] == "old output"
            return (
                '{"type":"plain","content":"compressed new output"}',
                CompressionAttempt(time_ms=5.0),
            )

        observation_strategy.compress.side_effect = compress
        agent.strategy = SlidingWindowCoACTCompression(
            observation_strategy=observation_strategy,
            sliding_window=SlidingWindow(window_size=1),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "tool",
                "content": "old output",
                "extra": {
                    "raw_output": "old output",
                    "returncode": 0,
                    "timestamp": 0.0,
                    "exception_info": None,
                },
            },
        ]
        agent.model = ObservationModel()
        agent.env = SimpleNamespace(
            execute=lambda action: {
                "output": "new raw output",
                "returncode": 0,
                "exception_info": None,
            }
        )
        agent.get_template_vars = lambda **kwargs: {}
        agent.logger = MagicMock()

        result = agent.execute_actions(
            {
                "extra": {
                    "actions": [
                        {
                            "command": "cat file.py",
                            "context_focus_question": "Inspect new file.",
                        }
                    ]
                }
            }
        )

        assert result[-1]["content"].endswith("compressed new output")
        assert agent.messages[1]["content"] == SlidingWindow.removed_content
        assert agent.messages[1]["extra"]["raw_output"] == SlidingWindow.removed_content
        assert agent.messages[-1]["content"].endswith("compressed new output")
        assert agent.messages[-1]["extra"]["original_tool_output"] == "new raw output"
        assert agent._compression_stats["total_time_ms"] == 5.0
        assert agent._compression_stats["compressed_steps"] == 1

    def test_agentdiet_coact_compresses_new_output_then_reduces_history(self) -> None:
        """AgentDiet + CoACT should compress observations before reducing history."""

        class ObservationModel:
            def format_observation_messages(self, message, outputs, template_vars):
                output = outputs[0]
                return [
                    {
                        "role": "tool",
                        "content": output["output"],
                        "extra": {
                            "raw_output": output["output"],
                            "returncode": output.get("returncode", 0),
                            "timestamp": 0.0,
                            "exception_info": output.get("exception_info"),
                        },
                    }
                ]

        observation_strategy = MagicMock()

        def compress(tool_output, context):
            assert tool_output == "new raw output"
            assert context["context_focus_question"] == ("Inspect new file.",)
            return (
                '{"type":"plain","content":"compressed new output"}',
                CompressionAttempt(time_ms=5.0),
            )

        observation_strategy.compress.side_effect = compress

        agentdiet_strategy = AgentDietCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8001/v1",
            delay_steps=1,
            window_before_steps=0,
            token_threshold=10,
        )
        agentdiet_strategy.reduce_message_in_history = (  # type: ignore[method-assign]
            lambda messages: AgentDietReductionResult(
                assistant_index=1,
                tool_index=2,
                reduced_assistant_content="reduced assistant",
                reduced_tool_content="reduced old output",
                attempt=CompressionAttempt(time_ms=9.0),
            )
        )

        agent = object.__new__(CompressingAgent)
        agent.strategy = AgentDietCoACTCompression(
            observation_strategy=observation_strategy,
            agentdiet_strategy=agentdiet_strategy,
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "assistant",
                "content": "first",
                "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": "old output",
                "extra": {
                    "raw_output": "old output",
                    "returncode": 0,
                    "timestamp": 0.0,
                    "exception_info": None,
                },
            },
        ]
        agent.model = ObservationModel()
        agent.env = SimpleNamespace(
            execute=lambda action: {
                "output": "new raw output",
                "returncode": 0,
                "exception_info": None,
            }
        )
        agent.get_template_vars = lambda **kwargs: {}
        agent.logger = MagicMock()

        result = agent.execute_actions(
            {
                "extra": {
                    "actions": [
                        {
                            "command": "cat file.py",
                            "context_focus_question": "Inspect new file.",
                        }
                    ]
                }
            }
        )

        assert result[-1]["content"].endswith("compressed new output")
        assert agent.messages[1]["content"] == "reduced assistant"
        assert agent.messages[2]["content"] == "reduced old output"
        assert agent.messages[-1]["content"].endswith("compressed new output")
        assert agent.messages[-1]["extra"]["original_tool_output"] == "new raw output"
        assert agent._compression_stats["total_time_ms"] == 14.0
        assert agent._compression_stats["compressed_steps"] == 2

    def test_unchanged_json_restores_original_output_and_still_counts_call_cost(self) -> None:
        """Unchanged JSON should preserve the output while still charging the model call."""
        agent = object.__new__(CompressingAgent)
        agent.strategy = MagicMock()
        agent.strategy.compress.return_value = (
            '{"type":"unchanged","content":null}',
            CompressionAttempt(
                time_ms=12.0,
                cost={
                    "input_cost_usd": 0.1,
                    "cache_read_cost_usd": 0.0,
                    "cache_creation_cost_usd": 0.0,
                    "output_cost_usd": 0.0,
                    "total_cost_usd": 0.1,
                },
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "total_tokens": 50,
                },
            ),
        )
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.model = None
        agent.messages = [{"role": "user", "content": "Fix the bug"}]

        outputs = [{"output": "exact original output"}]
        assistant_message = {
            "reasoning_content": "Inspect and patch files.",
            "extra": {
                "actions": [
                    {
                        "command": "cat a.py",
                        "context_focus_question": "Inspect the relevant code in a.py.",
                    }
                ]
            },
        }

        compressed_outputs = agent._compress_outputs(outputs, assistant_message)

        assert compressed_outputs[0]["output"] == "exact original output"
        assert agent._compression_stats["total_time_ms"] == 12.0
        assert agent._compression_stats["compressed_steps"] == 0
        assert agent._compression_stats["total_cost_usd"] == pytest.approx(0.1)
        assert agent._compression_stats["total_tokens"] == 50

    def test_agentdiet_reduces_delayed_tool_message(self) -> None:
        """AgentDiet branch should rewrite one delayed tool output in history."""

        class ObservationModel:
            def format_observation_messages(self, message, outputs, template_vars):
                output = outputs[0]
                return [
                    {
                        "role": "tool",
                        "content": output["output"],
                        "extra": {
                            "raw_output": output["output"],
                            "returncode": output.get("returncode", 0),
                            "timestamp": 0.0,
                            "exception_info": output.get("exception_info"),
                        },
                    }
                ]

        strategy = AgentDietCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8001/v1",
            delay_steps=1,
            window_before_steps=0,
            token_threshold=10,
        )
        strategy.reduce_message_in_history = (  # type: ignore[method-assign]
            lambda messages: AgentDietReductionResult(
                assistant_index=1,
                tool_index=2,
                reduced_assistant_content="reduced assistant",
                reduced_tool_content="reduced old output",
                attempt=CompressionAttempt(
                    time_ms=9.0,
                    cost={
                        "input_cost_usd": 0.1,
                        "cache_read_cost_usd": 0.01,
                        "cache_creation_cost_usd": 0.0,
                        "output_cost_usd": 0.02,
                        "total_cost_usd": 0.13,
                    },
                    usage={
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "cached_tokens": 4,
                        "cache_creation_input_tokens": 0,
                        "total_tokens": 150,
                    },
                ),
                agent_erased='<step id="1">\n<think>original assistant</think>\n<result>old output</result>\n</step>',
            )
        )

        agent = object.__new__(CompressingAgent)
        agent.strategy = strategy
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "assistant",
                "content": "first",
                "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": "old output",
                "extra": {
                    "raw_output": "old output",
                    "returncode": 0,
                    "timestamp": 0.0,
                    "exception_info": None,
                },
            },
        ]
        agent.model = ObservationModel()
        agent.env = SimpleNamespace(
            execute=lambda action: {
                "output": "new output",
                "returncode": 0,
                "exception_info": None,
            }
        )
        agent.get_template_vars = lambda **kwargs: {}
        agent.logger = MagicMock()

        result = agent.execute_actions({"extra": {"actions": [{"command": "cat file.py"}]}})

        assert result[-1]["content"] == "new output"
        assert agent.messages[1]["content"] == "reduced assistant"
        assert agent.messages[2]["content"] == "reduced old output"
        assert agent.messages[1]["extra"]["agentdiet_reduced"] is True
        assert agent.messages[2]["extra"]["agentdiet_reduced"] is True
        assert "original assistant" in agent.messages[1]["agent_erased"]
        assert agent._compression_stats["compressed_steps"] == 1
        assert agent._compression_stats["total_time_ms"] == 9.0
        assert agent._compression_stats["total_cost_usd"] == 0.13
        assert agent._compression_stats["prompt_tokens"] == 120
        assert agent._compression_stats["completion_tokens"] == 30
        assert agent._compression_stats["cached_tokens"] == 4
        assert agent._compression_stats["total_tokens"] == 150

    def test_agentdiet_rejected_reduction_still_counts_cost(self) -> None:
        """AgentDiet should count reduction time even when nothing is applied."""

        strategy = AgentDietCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8001/v1",
            delay_steps=1,
            window_before_steps=0,
            token_threshold=10,
        )
        strategy.reduce_message_in_history = (  # type: ignore[method-assign]
            lambda messages: AgentDietReductionResult(
                assistant_index=None,
                tool_index=None,
                reduced_assistant_content=None,
                reduced_tool_content=None,
                attempt=CompressionAttempt(
                    time_ms=7.5,
                    cost={
                        "input_cost_usd": 0.08,
                        "cache_read_cost_usd": 0.0,
                        "cache_creation_cost_usd": 0.0,
                        "output_cost_usd": 0.01,
                        "total_cost_usd": 0.09,
                    },
                    usage={
                        "prompt_tokens": 90,
                        "completion_tokens": 5,
                        "cached_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "total_tokens": 95,
                    },
                ),
            )
        )

        agent = object.__new__(CompressingAgent)
        agent.strategy = strategy
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "assistant",
                "content": "first",
                "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": "old output",
                "extra": {
                    "raw_output": "old output",
                    "returncode": 0,
                    "timestamp": 0.0,
                    "exception_info": None,
                },
            },
        ]
        agent.model = SimpleNamespace(
            format_observation_messages=lambda message, outputs, template_vars: [
                {
                    "role": "tool",
                    "content": outputs[0]["output"],
                    "extra": {
                        "raw_output": outputs[0]["output"],
                        "returncode": outputs[0].get("returncode", 0),
                        "timestamp": 0.0,
                        "exception_info": outputs[0].get("exception_info"),
                    },
                }
            ]
        )
        agent.env = SimpleNamespace(
            execute=lambda action: {
                "output": "new output",
                "returncode": 0,
                "exception_info": None,
            }
        )
        agent.get_template_vars = lambda **kwargs: {}
        agent.logger = MagicMock()

        result = agent.execute_actions({"extra": {"actions": [{"command": "cat file.py"}]}})

        assert result[-1]["content"] == "new output"
        assert agent.messages[1]["content"] == "first"
        assert agent.messages[2]["content"] == "old output"
        assert agent._compression_stats["compressed_steps"] == 0
        assert agent._compression_stats["total_time_ms"] == 7.5
        assert agent._compression_stats["total_cost_usd"] == 0.09
        assert agent._compression_stats["prompt_tokens"] == 90
        assert agent._compression_stats["completion_tokens"] == 5
        assert agent._compression_stats["total_tokens"] == 95

    def test_agentdiet_reduce_message_in_history_uses_real_reduction_path(self) -> None:
        """AgentDiet should reduce one delayed step through its real reflection path."""
        strategy = AgentDietCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8001/v1",
            delay_steps=1,
            window_before_steps=0,
            token_threshold=10,
            token_count_model="openai/Qwen3.5-4B",
        )
        reduced_step = "\n".join(
            [
                '<step id="1">',
                "<think>Shortened reasoning</think>",
                '<call tool="bash">{"command": "cat old.py"}</call>',
                "<result>reduced old output</result>",
                "</step>",
            ]
        )

        def fake_token_counter(*, model: str, text: str) -> int:
            del model
            if "reduced old output" in text:
                return 20
            if "old output" in text:
                return 100
            return 60

        with patch("src.eval.baselines.agentdiet.token_counter", side_effect=fake_token_counter):
            with patch(
                "src.eval.baselines.agentdiet.completion",
                return_value=_FakeCompletionResponse(reduced_step),
            ):
                reduction = strategy.reduce_message_in_history(_agentdiet_history())

        assert reduction.assistant_index == 1
        assert reduction.tool_index == 2
        assert reduction.reduced_assistant_content == (
            "(System reminder: compressed for better efficiency) Shortened reasoning"
        )
        parsed_tool = json.loads(reduction.reduced_tool_content or "{}")
        assert parsed_tool["output"] == "reduced old output"
        assert reduction.attempt.usage["prompt_tokens"] == 120
        assert reduction.attempt.usage["completion_tokens"] == 30
        assert reduction.attempt.usage["cached_tokens"] == 4
        assert reduction.agent_erased is not None
        assert "old output" in reduction.agent_erased

    def test_agentdiet_rejects_reduction_with_savings_below_threshold(self) -> None:
        """Per AgentDiet paper: accept only if l_orig - l_reduced > theta."""
        strategy = AgentDietCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8001/v1",
            delay_steps=1,
            window_before_steps=0,
            token_threshold=50,
            token_count_model="openai/Qwen3.5-4B",
        )
        reduced_step = "\n".join(
            [
                '<step id="1">',
                "<think>Slightly shorter reasoning</think>",
                '<call tool="bash">{"command": "cat old.py"}</call>',
                "<result>marginally reduced old output</result>",
                "</step>",
            ]
        )

        # Original step token count: 100. Reduced step: 80. Savings: 20.
        # token_threshold=50, so 20 <= 50 should reject the reduction even
        # though the rewrite is technically shorter.
        def fake_token_counter(*, model: str, text: str) -> int:
            del model
            if "marginally reduced" in text:
                return 80
            if "old output" in text:
                return 100
            return 60

        with patch("src.eval.baselines.agentdiet.token_counter", side_effect=fake_token_counter):
            with patch(
                "src.eval.baselines.agentdiet.completion",
                return_value=_FakeCompletionResponse(reduced_step),
            ):
                reduction = strategy.reduce_message_in_history(_agentdiet_history())

        # Reduction was rejected: indices and reduced contents are None,
        # but the LLM call still happened so the attempt carries cost/usage.
        assert reduction.assistant_index is None
        assert reduction.tool_index is None
        assert reduction.reduced_assistant_content is None
        assert reduction.reduced_tool_content is None
        assert reduction.attempt.usage["prompt_tokens"] == 120
        assert reduction.attempt.usage["completion_tokens"] == 30

    def test_agentdiet_reduce_message_in_history_propagates_token_counter_error(self) -> None:
        """AgentDiet token counting errors should surface instead of silently falling back."""
        strategy = AgentDietCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8001/v1",
            delay_steps=1,
            window_before_steps=0,
            token_threshold=10,
            token_count_model="openai/Qwen3.5-35B-A3B-FP8",
        )

        with patch(
            "src.eval.baselines.agentdiet.token_counter",
            side_effect=RuntimeError("unknown token counter model"),
        ):
            with pytest.raises(RuntimeError, match="unknown token counter model"):
                strategy.reduce_message_in_history(_agentdiet_history())

    def test_swepruner_branch_bypasses_response_resolution(self) -> None:
        """SWEPruner output carries paper feedback prefix; envelope helpers must be skipped."""
        original_output = "line A\n" * 200
        pruned_output = "line A\n(filtered 198 lines)\nline A"

        strategy = SWEPrunerCompression(
            endpoint="http://swepruner",
            retries=1,
            skip_compression_max_tokens=4,
        )

        prune_response = MagicMock()
        prune_response.raise_for_status = MagicMock()
        prune_response.json.return_value = {
            "score": 0.88,
            "pruned_code": pruned_output,
            "kept_frags": [1, 200],
            "origin_token_cnt": 800,
            "left_token_cnt": 12,
            "model_input_token_cnt": 803,
            "error_msg": None,
        }

        agent = object.__new__(CompressingAgent)
        agent.strategy = strategy
        agent.compression_filter = None
        agent._compression_stats = dict(COMPRESSION_STATS_DEFAULTS)
        agent.messages = [{"role": "user", "content": "Find the bug"}]
        agent.logger = MagicMock()
        agent.model = MagicMock()
        agent.env = MagicMock()
        agent.env.execute.return_value = {
            "output": original_output,
            "returncode": 0,
            "exception_info": None,
        }
        agent.get_template_vars = lambda **kwargs: {}

        captured_outputs: list[list[dict]] = []

        def format_observation_messages(message, outputs, template_vars):
            captured_outputs.append([dict(o) for o in outputs])
            return [{"role": "tool", "content": outputs[0]["output"]}]

        agent.model.format_observation_messages = format_observation_messages

        message = {
            "extra": {
                "actions": [
                    {
                        "command": "cat huge_file.py",
                        "context_focus_question": "What's wrong with the function?",
                    }
                ]
            }
        }

        with (
            patch.object(strategy._client, "post", return_value=prune_response) as mock_post,
            patch("src.eval.compressing_agent.interpret_compression_response") as mock_interpret,
        ):
            result = agent.execute_actions(message)

        mock_post.assert_called_once()
        mock_interpret.assert_not_called()

        delivered_output = captured_outputs[0][0]["output"]
        # Paper-aligned envelope, not the runner's [Original tool output has been compressed]
        # annotation (which is now skipped for SWEPruner since it owns its own prefix).
        assert delivered_output.startswith(
            "Filtered some unrelevant parts judged by your context_focus_question, "
            "good try! Filtered Output:\n"
        )
        assert pruned_output in delivered_output
        assert "[Original tool output has been compressed]" not in delivered_output
        assert result == [{"role": "tool", "content": delivered_output}]

        stats = agent._compression_stats
        assert stats["compressed_steps"] == 1
        assert stats["total_time_ms"] > 0.0
        assert stats["prompt_tokens"] == 800
        assert stats["completion_tokens"] == 12
        assert stats["total_tokens"] == 812
        assert stats["total_cost_usd"] == 0.0
