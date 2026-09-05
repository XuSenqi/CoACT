"""Tests for the CoACT HTTP prune service core logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from src.serving.coact_prune import (
    CoACTBackendConfig,
    PruneRequest,
    check_backend_reachable,
    compute_kept_frags,
    count_tokens,
    prune_code,
)


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()

    def _encode(text: str, add_special_tokens: bool = False, return_attention_mask: bool = False):
        del add_special_tokens, return_attention_mask
        if not text:
            return {"input_ids": []}
        return {"input_ids": list(range(len(text.split())))}

    tokenizer.side_effect = _encode
    return tokenizer


def _vllm_response(*, content: str, prompt_tokens: int = 120) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 12},
    }
    return response


@pytest.fixture
def backend_config() -> CoACTBackendConfig:
    return CoACTBackendConfig(backend="http://localhost:8001/v1", model="CoACT")


@pytest.fixture
def tokenizer() -> MagicMock:
    return _make_tokenizer()


def test_count_tokens_uses_tokenizer(tokenizer: MagicMock) -> None:
    assert count_tokens("one two three", tokenizer) == 3
    assert count_tokens("", tokenizer) == 0


def test_compute_kept_frags_for_unchanged() -> None:
    original = "alpha\nbeta\ngamma"
    kept = compute_kept_frags(original, original, "unchanged")
    assert kept == [1, 2, 3]


def test_compute_kept_frags_for_code() -> None:
    original = "alpha\nbeta\ngamma"
    pruned = "alpha\n(compressed 1 lines: beta)\ngamma"
    kept = compute_kept_frags(original, pruned, "code")
    assert kept == [1, 3]


def test_prune_code_unchanged(backend_config: CoACTBackendConfig, tokenizer: MagicMock) -> None:
    client = MagicMock()
    client.post.return_value = _vllm_response(content='{"type":"unchanged","content":null}')

    response = prune_code(
        PruneRequest(query="find beta", code="alpha\nbeta\ngamma"),
        config=backend_config,
        tokenizer=tokenizer,
        client=client,
    )

    assert response.compression_type == "unchanged"
    assert response.pruned_code == "alpha\nbeta\ngamma"
    assert response.error_msg is None
    assert response.origin_token_cnt == response.left_token_cnt == 3
    assert response.model_input_token_cnt == 120
    assert response.kept_frags == [1, 2, 3]

    posted_payload = client.post.call_args.kwargs["json"]
    assert posted_payload["model"] == "CoACT"
    assert posted_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "find beta" in posted_payload["messages"][0]["content"]
    assert "1> alpha" in posted_payload["messages"][0]["content"]


def test_prune_code_plain(backend_config: CoACTBackendConfig, tokenizer: MagicMock) -> None:
    client = MagicMock()
    client.post.return_value = _vllm_response(
        content='{"type":"plain","content":"beta only"}',
        prompt_tokens=90,
    )

    response = prune_code(
        PruneRequest(query="find beta", code="alpha\nbeta\ngamma"),
        config=backend_config,
        tokenizer=tokenizer,
        client=client,
    )

    assert response.compression_type == "plain"
    assert response.pruned_code == "beta only"
    assert response.origin_token_cnt == 3
    assert response.left_token_cnt == 2
    assert response.kept_frags == []


def test_prune_code_code(backend_config: CoACTBackendConfig, tokenizer: MagicMock) -> None:
    client = MagicMock()
    client.post.return_value = _vllm_response(
        content='{"type":"code","content":["2:beta line"]}',
        prompt_tokens=150,
    )

    response = prune_code(
        PruneRequest(
            query="find alpha and gamma",
            code="alpha\nbeta\ngamma",
            goal="Fix the bug",
            tool_call="cat file.py",
        ),
        config=backend_config,
        tokenizer=tokenizer,
        client=client,
    )

    assert response.compression_type == "code"
    assert response.pruned_code == "alpha\n(compressed 1 lines: beta line)\ngamma"
    assert response.origin_token_cnt == 3
    assert response.left_token_cnt == 7
    assert response.kept_frags == [1, 3]

    prompt = client.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "Fix the bug" in prompt
    assert "$ cat file.py" in prompt


def test_prune_code_backend_failure_returns_original(
    backend_config: CoACTBackendConfig,
    tokenizer: MagicMock,
) -> None:
    client = MagicMock()
    client.post.side_effect = httpx.ConnectError("connection refused")

    response = prune_code(
        PruneRequest(query="find beta", code="alpha\nbeta"),
        config=backend_config,
        tokenizer=tokenizer,
        client=client,
    )

    assert response.compression_type == "invalid"
    assert response.pruned_code == "alpha\nbeta"
    assert response.error_msg == "connection refused"
    assert response.model_input_token_cnt == 0


def test_prune_code_invalid_model_json_falls_back(
    backend_config: CoACTBackendConfig,
    tokenizer: MagicMock,
) -> None:
    client = MagicMock()
    client.post.return_value = _vllm_response(content="not valid json")

    response = prune_code(
        PruneRequest(query="find beta", code="alpha\nbeta"),
        config=backend_config,
        tokenizer=tokenizer,
        client=client,
    )

    assert response.compression_type == "invalid"
    assert response.pruned_code == "alpha\nbeta"


def test_check_backend_reachable_success(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()

    def _fake_get(url: str, timeout: float = 5.0):
        del timeout
        assert url.endswith("/models")
        return response

    monkeypatch.setattr("src.serving.coact_prune.httpx.get", _fake_get)
    assert check_backend_reachable("http://localhost:8001/v1") is True


def test_check_backend_reachable_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(url: str, timeout: float = 5.0):
        del url, timeout
        raise httpx.ConnectError("down")

    monkeypatch.setattr("src.serving.coact_prune.httpx.get", _fake_get)
    assert check_backend_reachable("http://localhost:8001/v1") is False
