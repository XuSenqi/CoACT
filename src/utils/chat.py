"""Chat template utilities for training data formatting."""

from transformers import PreTrainedTokenizerBase


def render_chat_prompt(prompt: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """Apply the chat template to a raw user prompt for training.

    Returns the formatted prompt string (not tokenized) with the generation
    prompt appended, ready to be concatenated with a completion string.

    Args:
        prompt: Raw user message content.
        tokenizer: Tokenizer whose chat template to apply.

    Returns:
        Formatted prompt text ending at the assistant turn start.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
