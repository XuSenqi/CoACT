"""Text similarity utilities.

Provides Levenshtein distance-based similarity computation and
deduplication functions for text samples.
"""

from collections.abc import Callable, Sequence
from typing import Any

from rapidfuzz.distance import Levenshtein


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity between two strings.

    Similarity = 1 - (edit_distance / max(len(s1), len(s2))).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Similarity score in range [0, 1], where 1 means identical.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    return float(Levenshtein.normalized_similarity(s1, s2))


def deduplicate_by_similarity(
    items: Sequence[Any],
    get_text: Callable[[Any], str],
    threshold: float,
) -> list[Any]:
    """Deduplicate items by text similarity.

    Iterates through items in order, keeping an item only if it's not
    too similar to any previously kept item.

    Args:
        items: Sequence of items to deduplicate. Original order is preserved.
        get_text: Function that extracts the text used for similarity comparison.
        threshold: Similarity threshold in [0, 1]. Items with similarity greater
            than or equal to the threshold are considered duplicates.

    Returns:
        Deduplicated list of items.

    Raises:
        ValueError: If threshold is outside [0, 1].
        TypeError: If get_text returns a non-string value.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    if not items:
        return []

    kept: list[Any] = []
    kept_texts: list[str] = []

    for item in items:
        text = get_text(item)
        if not isinstance(text, str):
            raise TypeError(f"get_text must return str, got {type(text).__name__}")

        is_duplicate = False
        for kept_text in kept_texts:
            similarity = Levenshtein.normalized_similarity(
                text,
                kept_text,
                score_cutoff=threshold,
            )
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(item)
            kept_texts.append(text)

    return kept
