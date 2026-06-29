"""Tests for compression strategy implementations."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.config.config import (
    AgentDietConfig,
    CoACTConfig,
    Config,
    EvaluationConfig,
    LLMLingua2Config,
    LongCodeZipConfig,
    SWEPrunerConfig,
)
from src.eval.baselines import (
    AgentDietCoACTCompression,
    AgentDietCompression,
    CoACTCompression,
    LLMLingua2Compression,
    LongCodeZipCompression,
    SlidingWindow,
    SlidingWindowCoACTCompression,
    SWEPrunerCompression,
    VanillaCompression,
    get_strategy,
)


class TestVanillaCompression:
    """Tests for VanillaCompression strategy."""

    def test_compress_returns_unchanged_output(self) -> None:
        """VanillaCompression should return the original output unchanged."""
        strategy = VanillaCompression()
        original = "This is a long tool output with lots of information."

        result, attempt = strategy.compress(original)

        assert result == original
        assert attempt is None

    def test_name_attribute(self) -> None:
        """VanillaCompression should have name 'vanilla'."""
        strategy = VanillaCompression()
        assert strategy.name == "vanilla"


class TestSlidingWindow:
    """Tests for SlidingWindow strategy."""

    def test_initial_state(self) -> None:
        """SlidingWindow should initialize with correct state."""
        strategy = SlidingWindow(window_size=10)

        assert strategy.window_size == 10

    def test_get_message_indices_to_remove_returns_absolute_positions(self) -> None:
        """Should return absolute indices in the full message history."""
        strategy = SlidingWindow(window_size=2)
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "tool 1"},
            {"role": "assistant", "content": "thinking 2"},
            {"role": "tool", "content": "tool 2"},
            {"role": "assistant", "content": "thinking 3"},
            {"role": "tool", "content": "tool 3"},
        ]

        indices = strategy.get_message_indices_to_remove(messages)

        assert indices == [2]

    def test_get_message_indices_to_remove_skips_already_redacted_messages(self) -> None:
        """Should only return newly overflowed tool messages."""
        strategy = SlidingWindow(window_size=2)
        messages = [
            {"role": "tool", "content": strategy.removed_content},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "tool 2"},
            {"role": "assistant", "content": "thinking 2"},
            {"role": "tool", "content": "tool 3"},
            {"role": "assistant", "content": "thinking 3"},
            {"role": "tool", "content": "tool 4"},
        ]

        indices = strategy.get_message_indices_to_remove(messages)

        assert indices == [2]

    def test_compress_returns_tuple(self) -> None:
        """compress should return tuple of (output, attempt)."""
        strategy = SlidingWindow(window_size=5)

        result, attempt = strategy.compress("output")

        assert result == "output"
        assert attempt is None

    def test_name_attribute(self) -> None:
        """SlidingWindow should have correct name."""
        strategy = SlidingWindow()
        assert strategy.name == "sliding_window"


class TestCoACTCompression:
    """Tests for CoACTCompression strategy."""

    def test_initialization(self) -> None:
        """CoACTCompression should initialize with correct parameters."""
        strategy = CoACTCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8002/v1",
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            min_p=0.15,
            presence_penalty=0.4,
            repetition_penalty=1.3,
            max_tokens=81920,
            timeout=50.0,
            max_retries=8,
        )

        assert strategy.model == "openai/Qwen3.5-4B"
        assert strategy.api_endpoint == "http://localhost:8002/v1"
        assert strategy.temperature == 0.6
        assert strategy.top_k == 50
        assert strategy.min_p == 0.15
        assert strategy.presence_penalty == 0.4
        assert strategy.repetition_penalty == 1.3
        assert strategy.timeout == 50.0
        assert strategy.max_retries == 8

    def test_from_config(self) -> None:
        """Should create CoACTCompression from Config."""
        config = Config(
            agent=MagicMock(request_timeout=21.0, model_retry_stop_after_attempt=5),
            evaluation=EvaluationConfig(
                coact=CoACTConfig(
                    model="openai/coact-model",
                    api_endpoint="http://localhost:8002/v1",
                    temperature=0.5,
                    top_p=0.9,
                    top_k=35,
                    min_p=0.07,
                    presence_penalty=0.25,
                    repetition_penalty=1.15,
                    max_tokens=2048,
                    request_timeout=21.0,
                    model_retry_stop_after_attempt=5,
                )
            ),
        )

        strategy = CoACTCompression.from_config(config)

        assert strategy.model == "openai/coact-model"
        assert strategy.api_endpoint == "http://localhost:8002/v1"
        assert strategy.temperature == 0.5
        assert strategy.top_k == 35
        assert strategy.min_p == 0.07
        assert strategy.presence_penalty == 0.25
        assert strategy.repetition_penalty == 1.15
        assert strategy.max_tokens == 2048
        assert strategy.timeout == 21.0
        assert strategy.max_retries == 5

    def test_compress_passes_timeout_and_retry_settings(self) -> None:
        """Compress should forward timeout and retry settings to LiteLLM."""
        strategy = CoACTCompression(
            model="openai/Qwen3.5-4B",
            api_endpoint="http://localhost:8002/v1",
            timeout=33.0,
            max_retries=6,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="compressed output"))]
        mock_response.model_dump.return_value = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        }

        with patch(
            "src.eval.baselines.model_compression.completion", return_value=mock_response
        ) as mock_completion:
            compressed, attempt = strategy.compress(
                "original output",
                {
                    "goal": "Fix bug",
                    "context_focus_question": ("Inspect output",),
                    "tool_call": ("cat file.py",),
                    "tool_output_for_prompt": "original output",
                },
            )

        assert compressed == "compressed output"
        assert attempt is not None
        assert attempt.usage["prompt_tokens"] == 10
        assert attempt.usage["completion_tokens"] == 5
        assert attempt.usage["total_tokens"] == 15
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["timeout"] == 33.0
        assert kwargs["max_retries"] == 6
        assert kwargs["top_k"] == 20
        assert kwargs["min_p"] == 0.0
        assert kwargs["presence_penalty"] == 0.0
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_name_attribute(self) -> None:
        """CoACTCompression should have name 'CoACT'."""
        strategy = CoACTCompression(
            model="test",
            api_endpoint="http://test",
        )
        assert strategy.name == "CoACT"


class TestSWEPrunerCompression:
    """Tests for SWEPrunerCompression strategy."""

    def test_initialization(self) -> None:
        """SWEPrunerCompression should retain constructor params."""
        strategy = SWEPrunerCompression(
            endpoint="http://test:9000",
            threshold=0.7,
            always_keep_first_frags=True,
            chunk_overlap_tokens=10,
            request_timeout=30.0,
            retries=5,
            skip_compression_max_tokens=256,
        )

        assert strategy.endpoint == "http://test:9000"
        assert strategy.threshold == 0.7
        assert strategy.always_keep_first_frags is True
        assert strategy.chunk_overlap_tokens == 10
        assert strategy.request_timeout == 30.0
        assert strategy.retries == 5
        assert strategy.skip_compression_max_tokens == 256

    def test_endpoint_trailing_slash_stripped(self) -> None:
        """Trailing slash on endpoint should be removed so /prune is appended cleanly."""
        strategy = SWEPrunerCompression(endpoint="http://x/")
        assert strategy.endpoint == "http://x"

    def test_from_config(self) -> None:
        """from_config should populate all knobs from EvaluationConfig.swepruner."""
        config = Config(
            agent=MagicMock(request_timeout=120.0, model_retry_stop_after_attempt=3),
            evaluation=EvaluationConfig(
                swepruner=SWEPrunerConfig(
                    endpoint="http://prune:7000",
                    threshold=0.6,
                    always_keep_first_frags=False,
                    chunk_overlap_tokens=64,
                    request_timeout=15.0,
                    retries=2,
                    skip_compression_max_tokens=400,
                ),
            ),
        )

        strategy = SWEPrunerCompression.from_config(config)

        assert strategy.endpoint == "http://prune:7000"
        assert strategy.threshold == 0.6
        assert strategy.always_keep_first_frags is False
        assert strategy.chunk_overlap_tokens == 64
        assert strategy.request_timeout == 15.0
        assert strategy.retries == 2
        assert strategy.skip_compression_max_tokens == 400

    def test_compress_short_outputs_get_all_kept_prefix_without_http_call(self) -> None:
        """Mirror paper's min_chars bypass: short input -> "All kept" envelope, no HTTP."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            skip_compression_max_tokens=100,
        )
        short_output = "tiny"

        with patch.object(strategy._client, "post") as mock_post:
            result, attempt = strategy.compress(
                short_output,
                context={"context_focus_question": ("anything",)},
            )

        assert result == "All outputs are judged as relevent! Output:\ntiny"
        assert attempt is None
        mock_post.assert_not_called()

    def test_compress_skips_when_focus_question_empty(self) -> None:
        """No focus question -> no prune request, original returned without any prefix."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            skip_compression_max_tokens=4,
        )
        large_output = "x" * 1024

        with patch.object(strategy._client, "post") as mock_post:
            result, attempt = strategy.compress(
                large_output,
                context={"context_focus_question": (None, "   ")},
            )

        assert result == large_output
        assert attempt is None
        mock_post.assert_not_called()

    def test_compress_calls_prune_endpoint_and_returns_attempt(self) -> None:
        """Compress should POST to /prune, prepend the Filtered envelope, and surface tokens."""
        strategy = SWEPrunerCompression(
            endpoint="http://x:1234",
            threshold=0.55,
            always_keep_first_frags=True,
            chunk_overlap_tokens=16,
            retries=1,
            skip_compression_max_tokens=4,
        )
        large_output = "line A\n" * 200
        pruned_code = "line A\n(filtered 198 lines)\nline A"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "score": 0.91,
            "pruned_code": pruned_code,
            "kept_frags": [1, 200],
            "origin_token_cnt": 800,
            "left_token_cnt": 12,
            "model_input_token_cnt": 803,
            "error_msg": None,
        }

        with patch.object(strategy._client, "post", return_value=mock_response) as mock_post:
            result, attempt = strategy.compress(
                large_output,
                context={"context_focus_question": ("focus on bug",)},
            )

        mock_post.assert_called_once()
        (url,) = mock_post.call_args.args
        assert url == "http://x:1234/prune"
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["code"] == large_output
        assert sent_payload["query"] == "focus on bug"
        assert sent_payload["threshold"] == 0.55
        assert sent_payload["always_keep_first_frags"] is True
        assert sent_payload["chunk_overlap_tokens"] == 16

        assert result.startswith(
            "Filtered some unrelevant parts judged by your context_focus_question, "
            "good try! Filtered Output:\n"
        )
        assert pruned_code in result
        assert attempt is not None
        assert attempt.time_ms >= 0.0
        assert attempt.usage["prompt_tokens"] == 800
        assert attempt.usage["completion_tokens"] == 12
        assert attempt.cost["total_cost_usd"] == 0.0
        assert attempt.extras["model"] == "swepruner"
        assert attempt.extras["origin_token_cnt"] == 800
        assert attempt.extras["left_token_cnt"] == 12
        assert attempt.extras["score"] == 0.91

    def test_compress_all_kept_branch_uses_all_kept_prefix(self) -> None:
        """When the service returns left == origin, agent gets the All-kept envelope."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            retries=1,
            skip_compression_max_tokens=4,
        )
        original = "fully relevant output\n" * 100

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "score": 0.99,
            "pruned_code": original,
            "kept_frags": list(range(1, 101)),
            "origin_token_cnt": 500,
            "left_token_cnt": 500,
            "model_input_token_cnt": 503,
            "error_msg": None,
        }

        with patch.object(strategy._client, "post", return_value=mock_response):
            result, attempt = strategy.compress(
                original,
                context={"context_focus_question": ("q",)},
            )

        assert result == "All outputs are judged as relevent! Output:\n" + original
        assert attempt is not None
        assert attempt.usage["prompt_tokens"] == 500
        assert attempt.usage["completion_tokens"] == 500

    def test_compress_returns_pruner_error_envelope_when_service_flags_error(self) -> None:
        """Service-level error_msg must wrap the original output in [Pruner Error] (no raise)."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            retries=1,
            skip_compression_max_tokens=4,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "score": 0.0,
            "pruned_code": "",
            "kept_frags": [],
            "origin_token_cnt": 0,
            "left_token_cnt": 0,
            "model_input_token_cnt": 0,
            "error_msg": "query too long",
        }

        with patch.object(strategy._client, "post", return_value=mock_response):
            result, attempt = strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("q",)},
            )

        assert result == ("[Pruner Error]: query too long\n\nOriginal Output:\n" + "x" * 100)
        assert attempt is not None
        assert attempt.extras["error_msg"] == "query too long"
        assert attempt.usage["prompt_tokens"] == 0  # not counted in stats
        assert attempt.usage["completion_tokens"] == 0

    def test_compress_retries_then_falls_back_when_all_attempts_fail(self) -> None:
        """All HTTP attempts failing -> [Pruner Error] envelope wrapping the original output."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            retries=3,
            skip_compression_max_tokens=4,
        )

        with patch.object(
            strategy._client,
            "post",
            side_effect=httpx.ConnectError("connection refused"),
        ) as mock_post:
            result, attempt = strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("q",)},
            )

        assert mock_post.call_count == 3
        assert result.startswith("[Pruner Error]: ")
        assert "connection refused" in result
        assert "Original Output:\n" + "x" * 100 in result
        assert attempt is not None
        assert attempt.extras["error_msg"]
        assert "connection refused" in attempt.extras["error_msg"]

    def test_compress_recovers_on_retry(self) -> None:
        """A transient failure followed by success returns the success body."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            retries=3,
            skip_compression_max_tokens=4,
        )

        success = MagicMock()
        success.raise_for_status = MagicMock()
        success.json.return_value = {
            "score": 0.7,
            "pruned_code": "kept",
            "kept_frags": [1],
            "origin_token_cnt": 10,
            "left_token_cnt": 5,
            "model_input_token_cnt": 12,
            "error_msg": None,
        }

        with patch.object(
            strategy._client,
            "post",
            side_effect=[httpx.ConnectError("flaky"), success],
        ) as mock_post:
            result, attempt = strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("q",)},
            )

        assert mock_post.call_count == 2
        assert result.endswith("kept")
        assert "Filtered Output:" in result
        assert attempt is not None
        assert attempt.usage["prompt_tokens"] == 10
        assert attempt.usage["completion_tokens"] == 5

    def test_compress_joins_multiple_focus_questions(self) -> None:
        """Multiple per-action focus questions should join with newline into one query."""
        strategy = SWEPrunerCompression(
            endpoint="http://x",
            skip_compression_max_tokens=4,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "score": 0.5,
            "pruned_code": "ok",
            "kept_frags": [],
            "origin_token_cnt": 1,
            "left_token_cnt": 1,
            "model_input_token_cnt": 1,
            "error_msg": None,
        }

        with patch.object(strategy._client, "post", return_value=mock_response) as mock_post:
            strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("a", None, " b ", "")},
            )

        sent = mock_post.call_args.kwargs["json"]
        assert sent["query"] == "a\nb"

    def test_name_attribute(self) -> None:
        """SWEPrunerCompression should have name 'swepruner'."""
        strategy = SWEPrunerCompression(endpoint="http://x")
        assert strategy.name == "swepruner"


class TestLLMLingua2Compression:
    """Tests for LLMLingua2Compression strategy."""

    def test_initialization(self) -> None:
        """LLMLingua2Compression should retain constructor params."""
        strategy = LLMLingua2Compression(
            endpoint="http://test:9000",
            rate=0.3,
            target_token=256,
            force_tokens=("\n",),
            force_reserve_digit=True,
            drop_consecutive=True,
            chunk_end_tokens=(".",),
            use_token_level_filter=False,
            use_context_level_filter=True,
            request_timeout=30.0,
            retries=5,
            skip_compression_max_tokens=256,
        )

        assert strategy.endpoint == "http://test:9000"
        assert strategy.rate == 0.3
        assert strategy.target_token == 256
        assert strategy.force_tokens == ["\n"]
        assert strategy.force_reserve_digit is True
        assert strategy.drop_consecutive is True
        assert strategy.chunk_end_tokens == ["."]
        assert strategy.use_token_level_filter is False
        assert strategy.use_context_level_filter is True
        assert strategy.request_timeout == 30.0
        assert strategy.retries == 5
        assert strategy.skip_compression_max_tokens == 256

    def test_endpoint_trailing_slash_stripped(self) -> None:
        """Trailing slash on endpoint should be removed so /compress is appended cleanly."""
        strategy = LLMLingua2Compression(endpoint="http://x/")
        assert strategy.endpoint == "http://x"

    def test_default_knobs(self) -> None:
        """Knobs mirror upstream compress_prompt_llmlingua2, except rate.

        ``rate`` defaults to 0.8 (keep ~80% of tokens): the upstream 0.5 default
        drops so many tokens that code/tool output becomes unusable fragments.
        Every other knob still matches the official signature defaults.
        """
        strategy = LLMLingua2Compression(endpoint="http://x")

        assert strategy.rate == 0.8
        assert strategy.target_token == -1
        assert strategy.force_tokens == []
        assert strategy.force_reserve_digit is False
        assert strategy.drop_consecutive is False
        assert strategy.chunk_end_tokens == [".", "\n"]
        assert strategy.use_token_level_filter is True
        assert strategy.use_context_level_filter is False

    def test_from_config(self) -> None:
        """from_config should populate all knobs from EvaluationConfig.llmlingua2."""
        config = Config(
            agent=MagicMock(request_timeout=120.0, model_retry_stop_after_attempt=3),
            evaluation=EvaluationConfig(
                llmlingua2=LLMLingua2Config(
                    endpoint="http://lingua:7000",
                    rate=0.4,
                    target_token=-1,
                    force_tokens=("\n", "?"),
                    force_reserve_digit=True,
                    drop_consecutive=False,
                    chunk_end_tokens=(".", "\n"),
                    use_token_level_filter=True,
                    use_context_level_filter=False,
                    request_timeout=15.0,
                    retries=2,
                    skip_compression_max_tokens=400,
                ),
            ),
        )

        strategy = LLMLingua2Compression.from_config(config)

        assert strategy.endpoint == "http://lingua:7000"
        assert strategy.rate == 0.4
        assert strategy.force_tokens == ["\n", "?"]
        assert strategy.force_reserve_digit is True
        assert strategy.request_timeout == 15.0
        assert strategy.retries == 2
        assert strategy.skip_compression_max_tokens == 400

    def test_name_and_flags(self) -> None:
        """LLMLingua-2 is task-agnostic (no CFQ) but announces compression to the agent."""
        strategy = LLMLingua2Compression(endpoint="http://x")
        assert strategy.name == "llmlingua2"
        assert strategy.consumes_cfq is False
        assert strategy.announces_compression is True

    def test_compress_short_output_skips_http(self) -> None:
        """Short inputs bypass compression entirely: original returned, no HTTP, no attempt."""
        strategy = LLMLingua2Compression(
            endpoint="http://x",
            skip_compression_max_tokens=100,
        )
        short_output = "tiny"

        with patch.object(strategy._client, "post") as mock_post:
            result, attempt = strategy.compress(short_output, context=None)

        assert result == short_output
        assert attempt is None
        mock_post.assert_not_called()

    def test_compress_calls_endpoint_and_returns_attempt(self) -> None:
        """Compress should POST to /compress, forward knobs, and surface token usage."""
        strategy = LLMLingua2Compression(
            endpoint="http://x:1234",
            rate=0.5,
            force_tokens=("\n",),
            retries=1,
            skip_compression_max_tokens=4,
        )
        large_output = "line A\n" * 200
        compressed = "line A line A line A"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compressed_prompt": compressed,
            "origin_tokens": 800,
            "compressed_tokens": 400,
            "rate": "50.0%",
            "ratio": "2.0x",
            "error_msg": None,
        }

        with patch.object(strategy._client, "post", return_value=mock_response) as mock_post:
            result, attempt = strategy.compress(large_output, context=None)

        mock_post.assert_called_once()
        (url,) = mock_post.call_args.args
        assert url == "http://x:1234/compress"
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["text"] == large_output
        assert sent_payload["rate"] == 0.5
        assert sent_payload["force_tokens"] == ["\n"]
        assert sent_payload["target_token"] == -1

        assert result == compressed
        assert attempt is not None
        assert attempt.time_ms >= 0.0
        assert attempt.usage["prompt_tokens"] == 800
        assert attempt.usage["completion_tokens"] == 400
        assert attempt.cost["total_cost_usd"] == 0.0
        assert attempt.extras["model"] == "llmlingua2"
        assert attempt.extras["origin_tokens"] == 800
        assert attempt.extras["compressed_tokens"] == 400
        assert attempt.extras["ratio"] == "2.0x"

    def test_compress_falls_back_to_original_when_service_flags_error(self) -> None:
        """Service-level error_msg must return the original output unchanged (fail-soft)."""
        strategy = LLMLingua2Compression(
            endpoint="http://x",
            retries=1,
            skip_compression_max_tokens=4,
        )
        large_output = "x" * 1024

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compressed_prompt": "",
            "origin_tokens": 0,
            "compressed_tokens": 0,
            "rate": "",
            "ratio": "",
            "error_msg": "model OOM",
        }

        with patch.object(strategy._client, "post", return_value=mock_response):
            result, attempt = strategy.compress(large_output, context=None)

        assert result == large_output
        assert attempt is not None
        assert attempt.extras["error_msg"] == "model OOM"

    def test_compress_retries_then_falls_back_when_all_attempts_fail(self) -> None:
        """When every HTTP attempt raises, the original output is returned (fail-soft)."""
        strategy = LLMLingua2Compression(
            endpoint="http://x",
            retries=3,
            skip_compression_max_tokens=4,
        )
        large_output = "y" * 1024

        with patch.object(
            strategy._client, "post", side_effect=httpx.ConnectError("refused")
        ) as mock_post:
            result, attempt = strategy.compress(large_output, context=None)

        assert mock_post.call_count == 3
        assert result == large_output
        assert attempt is not None
        assert "refused" in attempt.extras["error_msg"]


class TestLongCodeZipCompression:
    """Tests for LongCodeZipCompression strategy."""

    def test_initialization(self) -> None:
        """LongCodeZipCompression should retain constructor params."""
        strategy = LongCodeZipCompression(
            endpoint="http://test:9000",
            rate=0.3,
            language="java",
            dynamic_compression_ratio=0.4,
            context_budget="+50",
            rank_only=True,
            fine_ratio=0.6,
            fine_grained_importance_method="contrastive_perplexity",
            min_lines_for_fine_grained=8,
            importance_beta=0.25,
            use_knapsack=False,
            request_timeout=30.0,
            retries=5,
            skip_compression_max_tokens=256,
        )

        assert strategy.endpoint == "http://test:9000"
        assert strategy.rate == 0.3
        assert strategy.language == "java"
        assert strategy.dynamic_compression_ratio == 0.4
        assert strategy.context_budget == "+50"
        assert strategy.rank_only is True
        assert strategy.fine_ratio == 0.6
        assert strategy.fine_grained_importance_method == "contrastive_perplexity"
        assert strategy.min_lines_for_fine_grained == 8
        assert strategy.importance_beta == 0.25
        assert strategy.use_knapsack is False
        assert strategy.request_timeout == 30.0
        assert strategy.retries == 5
        assert strategy.skip_compression_max_tokens == 256

    def test_endpoint_trailing_slash_stripped(self) -> None:
        """Trailing slash on endpoint should be removed so /compress is appended cleanly."""
        strategy = LongCodeZipCompression(endpoint="http://x/")
        assert strategy.endpoint == "http://x"

    def test_default_knobs_mirror_upstream(self) -> None:
        """Defaults mirror the upstream compress_code_file signature (full two-stage)."""
        strategy = LongCodeZipCompression(endpoint="http://x")

        assert strategy.rate == 0.5
        assert strategy.language == "python"
        assert strategy.dynamic_compression_ratio == 0.2
        assert strategy.context_budget == "+100"
        assert strategy.rank_only is False
        assert strategy.fine_ratio is None
        assert strategy.fine_grained_importance_method == "conditional_ppl"
        assert strategy.min_lines_for_fine_grained == 5
        assert strategy.importance_beta == 0.5
        assert strategy.use_knapsack is True

    def test_from_config(self) -> None:
        """from_config should populate all knobs from EvaluationConfig.longcodezip."""
        config = Config(
            agent=MagicMock(request_timeout=120.0, model_retry_stop_after_attempt=3),
            evaluation=EvaluationConfig(
                longcodezip=LongCodeZipConfig(
                    endpoint="http://zip:7000",
                    rate=0.4,
                    language="python",
                    dynamic_compression_ratio=0.3,
                    context_budget="+0",
                    rank_only=True,
                    fine_ratio=0.8,
                    fine_grained_importance_method="contrastive_perplexity",
                    min_lines_for_fine_grained=6,
                    importance_beta=0.7,
                    use_knapsack=False,
                    request_timeout=15.0,
                    retries=2,
                    skip_compression_max_tokens=400,
                ),
            ),
        )

        strategy = LongCodeZipCompression.from_config(config)

        assert strategy.endpoint == "http://zip:7000"
        assert strategy.rate == 0.4
        assert strategy.dynamic_compression_ratio == 0.3
        assert strategy.context_budget == "+0"
        assert strategy.rank_only is True
        assert strategy.fine_ratio == 0.8
        assert strategy.fine_grained_importance_method == "contrastive_perplexity"
        assert strategy.min_lines_for_fine_grained == 6
        assert strategy.importance_beta == 0.7
        assert strategy.use_knapsack is False
        assert strategy.request_timeout == 15.0
        assert strategy.retries == 2
        assert strategy.skip_compression_max_tokens == 400

    def test_name_and_flags(self) -> None:
        """LongCodeZip is query-aware (consumes CFQ) and does not announce compression."""
        strategy = LongCodeZipCompression(endpoint="http://x")
        assert strategy.name == "longcodezip"
        assert strategy.consumes_cfq is True
        assert strategy.announces_compression is False

    def test_compress_requires_context(self) -> None:
        """LongCodeZip is query-aware: a missing context is a hard error."""
        strategy = LongCodeZipCompression(endpoint="http://x")
        with pytest.raises(ValueError, match="context is required"):
            strategy.compress("x" * 1024, context=None)

    def test_compress_skips_when_focus_question_empty(self) -> None:
        """No focus question -> no compress request, original returned untouched."""
        strategy = LongCodeZipCompression(endpoint="http://x", skip_compression_max_tokens=4)
        large_output = "x" * 1024

        with patch.object(strategy._client, "post") as mock_post:
            result, attempt = strategy.compress(
                large_output,
                context={"context_focus_question": (None, "   ")},
            )

        assert result == large_output
        assert attempt is None
        mock_post.assert_not_called()

    def test_compress_short_output_skips_http(self) -> None:
        """Short inputs bypass compression entirely: original returned, no HTTP, no attempt."""
        strategy = LongCodeZipCompression(endpoint="http://x", skip_compression_max_tokens=100)
        short_output = "tiny"

        with patch.object(strategy._client, "post") as mock_post:
            result, attempt = strategy.compress(
                short_output,
                context={"context_focus_question": ("anything",)},
            )

        assert result == short_output
        assert attempt is None
        mock_post.assert_not_called()

    def test_compress_calls_endpoint_and_returns_attempt(self) -> None:
        """Compress should POST to /compress, forward knobs, and surface token usage."""
        strategy = LongCodeZipCompression(
            endpoint="http://x:1234",
            rate=0.4,
            rank_only=False,
            importance_beta=0.5,
            retries=1,
            skip_compression_max_tokens=4,
        )
        large_output = "def f():\n    return 1\n" * 200

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compressed_code": "def f():\n    return 1",
            "original_tokens": 800,
            "compressed_tokens": 400,
            "compression_ratio": 2.0,
            "error_msg": None,
        }

        with patch.object(strategy._client, "post", return_value=mock_response) as mock_post:
            result, attempt = strategy.compress(
                large_output,
                context={"context_focus_question": ("fix the bug",)},
            )

        mock_post.assert_called_once()
        (url,) = mock_post.call_args.args
        assert url == "http://x:1234/compress"
        sent = mock_post.call_args.kwargs["json"]
        assert sent["code"] == large_output
        assert sent["query"] == "fix the bug"
        assert sent["rate"] == 0.4
        assert sent["rank_only"] is False
        assert sent["use_knapsack"] is True

        assert result == "def f():\n    return 1"
        assert attempt is not None
        assert attempt.time_ms >= 0.0
        assert attempt.usage["prompt_tokens"] == 800
        assert attempt.usage["completion_tokens"] == 400
        assert attempt.cost["total_cost_usd"] == 0.0
        assert attempt.extras["model"] == "longcodezip"
        assert attempt.extras["compression_ratio"] == 2.0

    def test_compress_joins_multiple_focus_questions(self) -> None:
        """Multiple per-action focus questions should join with newline into one query."""
        strategy = LongCodeZipCompression(endpoint="http://x", skip_compression_max_tokens=4)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compressed_code": "ok",
            "original_tokens": 1,
            "compressed_tokens": 1,
            "compression_ratio": 1.0,
            "error_msg": None,
        }

        with patch.object(strategy._client, "post", return_value=mock_response) as mock_post:
            strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("a", None, " b ", "")},
            )

        sent = mock_post.call_args.kwargs["json"]
        assert sent["query"] == "a\nb"

    def test_compress_falls_back_when_service_flags_error(self) -> None:
        """Service-level error_msg must return the original output (fail-soft, no raise)."""
        strategy = LongCodeZipCompression(
            endpoint="http://x",
            retries=1,
            skip_compression_max_tokens=4,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "compressed_code": "",
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_ratio": 1.0,
            "error_msg": "model OOM",
        }

        with patch.object(strategy._client, "post", return_value=mock_response):
            result, attempt = strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("q",)},
            )

        assert result == "x" * 100
        assert attempt is not None
        assert attempt.extras["error_msg"] == "model OOM"

    def test_compress_retries_then_falls_back_when_all_attempts_fail(self) -> None:
        """When every HTTP attempt raises, the original output is returned (fail-soft)."""
        strategy = LongCodeZipCompression(
            endpoint="http://x",
            retries=3,
            skip_compression_max_tokens=4,
        )

        with patch.object(
            strategy._client, "post", side_effect=httpx.ConnectError("refused")
        ) as mock_post:
            result, attempt = strategy.compress(
                "x" * 100,
                context={"context_focus_question": ("q",)},
            )

        assert mock_post.call_count == 3
        assert result == "x" * 100
        assert attempt is not None
        assert "refused" in attempt.extras["error_msg"]


class TestGetStrategy:
    """Tests for get_strategy factory function."""

    @pytest.fixture
    def mock_config(self) -> Config:
        """Create a mock Config for testing."""
        return Config(
            agent=MagicMock(request_timeout=120.0, model_retry_stop_after_attempt=3),
            evaluation=EvaluationConfig(
                sliding_window_size=8,
                coact=CoACTConfig(
                    model="coact-model",
                    api_endpoint="http://localhost:8002/v1",
                    request_timeout=120.0,
                    model_retry_stop_after_attempt=3,
                ),
                agentdiet=AgentDietConfig(
                    model="agentdiet-model",
                    api_endpoint="http://localhost:8003/v1",
                    request_timeout=120.0,
                    model_retry_stop_after_attempt=3,
                    delay_steps=2,
                    window_before_steps=1,
                    token_threshold=500,
                ),
                swepruner=SWEPrunerConfig(
                    endpoint="http://localhost:8888",
                    threshold=0.4,
                    always_keep_first_frags=True,
                    chunk_overlap_tokens=32,
                    request_timeout=45.0,
                    retries=2,
                    skip_compression_max_tokens=128,
                ),
            ),
        )

    def test_get_vanilla_strategy(self, mock_config: Config) -> None:
        """Should return VanillaCompression for 'vanilla'."""
        strategy = get_strategy("vanilla", mock_config)

        assert isinstance(strategy, VanillaCompression)

    def test_get_sliding_window_strategy(self, mock_config: Config) -> None:
        """Should return SlidingWindow with config params for 'sliding_window'."""
        strategy = get_strategy("sliding_window", mock_config)

        assert isinstance(strategy, SlidingWindow)
        assert strategy.window_size == 8

    def test_get_coact_strategy(self, mock_config: Config) -> None:
        """Should return CoACTCompression for 'CoACT'."""
        strategy = get_strategy("CoACT", mock_config)

        assert isinstance(strategy, CoACTCompression)
        assert strategy.model == "coact-model"
        assert strategy.api_endpoint == "http://localhost:8002/v1"
        assert strategy.timeout == 120.0
        assert strategy.max_retries == 3

    def test_get_sliding_window_coact_strategy(self, mock_config: Config) -> None:
        """Should combine CoACT observation compression with sliding-window history redaction."""
        strategy = get_strategy("sliding_window_CoACT", mock_config)

        assert isinstance(strategy, SlidingWindowCoACTCompression)
        assert strategy.name == "sliding_window_CoACT"
        assert isinstance(strategy.observation_strategy, CoACTCompression)
        assert isinstance(strategy.sliding_window, SlidingWindow)
        assert strategy.observation_strategy.model == "coact-model"
        assert strategy.sliding_window.window_size == 8

    def test_get_agentdiet_strategy(self, mock_config: Config) -> None:
        """Should return AgentDietCompression for 'agentdiet'."""
        strategy = get_strategy("agentdiet", mock_config)

        assert isinstance(strategy, AgentDietCompression)
        assert strategy.model == "agentdiet-model"
        assert strategy.api_endpoint == "http://localhost:8003/v1"
        assert strategy.delay_steps == 2
        assert strategy.window_before_steps == 1
        assert strategy.token_threshold == 500

    def test_get_agentdiet_coact_strategy(self, mock_config: Config) -> None:
        """Should combine CoACT observation compression with AgentDiet history reduction."""
        strategy = get_strategy("agentdiet_CoACT", mock_config)

        assert isinstance(strategy, AgentDietCoACTCompression)
        assert strategy.name == "agentdiet_CoACT"
        assert isinstance(strategy.observation_strategy, CoACTCompression)
        assert isinstance(strategy.agentdiet_strategy, AgentDietCompression)
        assert strategy.observation_strategy.model == "coact-model"
        assert strategy.agentdiet_strategy.model == "agentdiet-model"

    def test_get_swepruner_strategy(self, mock_config: Config) -> None:
        """Should return SWEPrunerCompression for 'swepruner'."""
        strategy = get_strategy("swepruner", mock_config)

        assert isinstance(strategy, SWEPrunerCompression)
        assert strategy.endpoint == "http://localhost:8888"
        assert strategy.threshold == 0.4
        assert strategy.always_keep_first_frags is True
        assert strategy.chunk_overlap_tokens == 32
        assert strategy.request_timeout == 45.0
        assert strategy.retries == 2
        assert strategy.skip_compression_max_tokens == 128

    def test_get_llmlingua2_strategy(self, mock_config: Config) -> None:
        """Should return LLMLingua2Compression for 'llmlingua2' (default config)."""
        strategy = get_strategy("llmlingua2", mock_config)

        assert isinstance(strategy, LLMLingua2Compression)
        assert strategy.endpoint == "http://localhost:8004"
        assert strategy.rate == 0.8
        assert strategy.consumes_cfq is False

    def test_get_longcodezip_strategy(self, mock_config: Config) -> None:
        """Should return LongCodeZipCompression for 'longcodezip' (default config)."""
        strategy = get_strategy("longcodezip", mock_config)

        assert isinstance(strategy, LongCodeZipCompression)
        assert strategy.endpoint == "http://localhost:8005"
        assert strategy.rate == 0.5
        assert strategy.consumes_cfq is True

    def test_get_unknown_strategy_raises(self, mock_config: Config) -> None:
        """Should raise ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown compression strategy"):
            get_strategy("unknown", mock_config)


class TestConsumesCfqFlag:
    """consumes_cfq drives whether the agent gets the CFQ-augmented prompt."""

    def test_cfq_consumers_are_coact_swepruner(self) -> None:
        assert CoACTCompression.consumes_cfq is True
        assert SlidingWindowCoACTCompression.consumes_cfq is True
        assert AgentDietCoACTCompression.consumes_cfq is True
        assert SWEPrunerCompression.consumes_cfq is True

    def test_vanilla_strategies_do_not_consume_cfq(self) -> None:
        assert VanillaCompression.consumes_cfq is False
        assert SlidingWindow.consumes_cfq is False
        # AgentDiet assumes a vanilla agent with no context-focus question.
        assert AgentDietCompression.consumes_cfq is False
        # LLMLingua-2 is task-agnostic and ignores the focus question.
        assert LLMLingua2Compression.consumes_cfq is False

    def test_only_llmlingua2_announces_compression(self) -> None:
        """announces_compression injects the transparent-compression notice."""
        # LLMLingua-2 compresses unconditionally without CFQ, so it tells the agent.
        assert LLMLingua2Compression.announces_compression is True
        # CFQ consumers already announce compression via the CFQ guidance.
        assert CoACTCompression.announces_compression is False
        assert SlidingWindowCoACTCompression.announces_compression is False
        assert AgentDietCoACTCompression.announces_compression is False
        assert SWEPrunerCompression.announces_compression is False
        # Vanilla strategies do not compress transparently.
        assert VanillaCompression.announces_compression is False
        assert SlidingWindow.announces_compression is False
        assert AgentDietCompression.announces_compression is False
