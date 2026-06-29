"""Tests for the LoRA merge script."""

import torch

import scripts.merge_lora as merge_lora


class _MockTokenizer:
    """Tokenizer stub exposing a configurable vocabulary length."""

    def __init__(self, vocab_size: int) -> None:
        """Initialize the tokenizer stub."""
        self._vocab_size = vocab_size

    def __len__(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self._vocab_size


class _MockEmbedding(torch.nn.Module):
    """Embedding stub with a mutable weight matrix."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        """Initialize the embedding weight."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(vocab_size, hidden_size))


class _MockModel(torch.nn.Module):
    """Model stub for embedding alignment tests."""

    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        """Initialize the model stub."""
        super().__init__()
        self.embed_tokens = _MockEmbedding(vocab_size, hidden_size)
        self.lm_head = _MockEmbedding(vocab_size, hidden_size)
        self.resize_calls: list[int] = []

    def get_input_embeddings(self) -> _MockEmbedding:
        """Return the input embedding module."""
        return self.embed_tokens

    def resize_token_embeddings(self, new_size: int) -> _MockEmbedding:
        """Mimic transformers resize_token_embeddings behavior."""
        hidden_size = self.embed_tokens.weight.shape[1]
        self.embed_tokens = _MockEmbedding(new_size, hidden_size)
        self.lm_head = _MockEmbedding(new_size, hidden_size)
        self.resize_calls.append(new_size)
        return self.embed_tokens


def test_resize_base_model_embeddings_resizes_when_vocab_mismatch() -> None:
    """Test resizing model embeddings to match the tokenizer vocabulary."""
    model = _MockModel(vocab_size=248320)
    tokenizer = _MockTokenizer(vocab_size=248078)

    resized = merge_lora.resize_base_model_embeddings(model, tokenizer)

    assert resized is True
    assert model.resize_calls == [248078]
    assert model.get_input_embeddings().weight.shape[0] == 248078
    assert model.lm_head.weight.shape[0] == 248078


def test_resize_base_model_embeddings_skips_when_already_aligned() -> None:
    """Test that no resize happens when model and tokenizer already match."""
    model = _MockModel(vocab_size=248078)
    tokenizer = _MockTokenizer(vocab_size=248078)

    resized = merge_lora.resize_base_model_embeddings(model, tokenizer)

    assert resized is False
    assert model.resize_calls == []
