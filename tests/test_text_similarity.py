"""Tests for text similarity utilities in src/utils/text_similarity.py."""

import pytest

from src.utils.text_similarity import deduplicate_by_similarity, levenshtein_similarity


class TestLevenshteinSimilarity:
    """Tests for levenshtein_similarity function."""

    def test_identical(self) -> None:
        assert levenshtein_similarity("abc", "abc") == 1.0

    def test_different(self) -> None:
        sim = levenshtein_similarity("abc", "xyz")
        assert sim == pytest.approx(0.0, rel=0.01)

    def test_one_empty(self) -> None:
        assert levenshtein_similarity("abc", "") == 0.0
        assert levenshtein_similarity("", "abc") == 0.0

    def test_both_empty(self) -> None:
        assert levenshtein_similarity("", "") == 1.0

    def test_one_char_diff(self) -> None:
        sim = levenshtein_similarity("abc", "abd")
        assert sim == pytest.approx(0.667, rel=0.01)

    def test_longer_string(self) -> None:
        sim = levenshtein_similarity("hello world", "hello worl")
        assert sim == pytest.approx(0.909, rel=0.01)


class TestDeduplicateBySimilarity:
    """Tests for deduplicate_by_similarity function."""

    def test_empty_list(self) -> None:
        assert deduplicate_by_similarity([], lambda x: x, 0.9) == []

    def test_single_item(self) -> None:
        items = ["item1"]
        result = deduplicate_by_similarity(items, lambda x: x, 0.9)
        assert result == ["item1"]

    def test_no_duplicates(self) -> None:
        items = ["apple", "banana", "cherry"]
        result = deduplicate_by_similarity(items, lambda x: x, 0.9)
        assert result == ["apple", "banana", "cherry"]

    def test_exact_duplicates(self) -> None:
        items = ["apple", "apple", "apple"]
        result = deduplicate_by_similarity(items, lambda x: x, 0.9)
        assert result == ["apple"]

    def test_near_duplicates_high_threshold(self) -> None:
        items = ["hello world", "hello worl"]  # similarity ~0.91
        result = deduplicate_by_similarity(items, lambda x: x, 0.9)
        assert result == ["hello world"]

    def test_near_duplicates_low_threshold(self) -> None:
        items = ["hello world", "hello worl"]  # similarity ~0.91
        result = deduplicate_by_similarity(items, lambda x: x, 0.95)
        assert result == ["hello world", "hello worl"]

    def test_preserves_order(self) -> None:
        items = ["first", "second", "third", "first"]
        result = deduplicate_by_similarity(items, lambda x: x, 0.9)
        assert result == ["first", "second", "third"]

    def test_with_objects(self) -> None:
        class Item:
            def __init__(self, text):
                self.text = text

        items = [Item("hello"), Item("hello world"), Item("hello")]
        result = deduplicate_by_similarity(items, lambda x: x.text, 0.9)
        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[1].text == "hello world"

    def test_multiple_similar_items(self) -> None:
        items = ["test", "test1", "test2", "test"]  # test/test similarity=1.0
        result = deduplicate_by_similarity(items, lambda x: x, 0.9)
        # "test" is kept first, "test1" is similar enough? similarity = 0.8
        # "test2" vs "test": similarity = 0.8, vs "test1": ~0.8
        assert "test" in result

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
            deduplicate_by_similarity(["item"], lambda x: x, 1.1)

    def test_non_string_text_raises(self) -> None:
        with pytest.raises(TypeError, match="get_text must return str"):
            deduplicate_by_similarity([1], lambda x: x, 0.9)
