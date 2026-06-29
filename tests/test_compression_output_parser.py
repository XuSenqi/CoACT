"""Tests for JSON compression output parsing."""

from src.compression.output_parser import interpret_compression_response


def _numbered_output() -> str:
    """Build a numbered prompt-visible tool output fixture (no headers)."""
    return "\n".join(
        [
            "1> alpha",
            "2> beta",
            "3> gamma",
            "4> delta",
            "5> epsilon",
        ]
    )


def test_interpret_compression_response_keeps_original_for_unchanged_json() -> None:
    """Valid unchanged JSON should preserve the original tool output."""
    result = interpret_compression_response(
        '{"type":"unchanged","content":null}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is True
    assert result.valid_response is True
    assert result.used_line_references is False
    assert result.effective_output == "raw original output"
    assert result.normalized_completion == '{"type":"unchanged","content":null}'


def test_interpret_compression_response_repairs_plain_json() -> None:
    """Repairable JSON should be normalized into canonical plain-output JSON."""
    result = interpret_compression_response(
        "{type:'plain', content:'short summary',}",
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is False
    assert result.valid_response is True
    assert result.used_line_references is False
    assert result.effective_output == "short summary"
    assert result.normalized_completion == '{"type":"plain","content":"short summary"}'


def test_interpret_compression_response_omits_single_lines() -> None:
    """Code mode should omit specified lines and keep the rest."""
    result = interpret_compression_response(
        '{"type":"code","content":["2:beta line","4:delta line"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is False
    assert result.valid_response is True
    assert result.used_line_references is True
    assert result.effective_output == "\n".join(
        [
            "alpha",
            "(compressed 1 lines: beta line)",
            "gamma",
            "(compressed 1 lines: delta line)",
            "epsilon",
        ]
    )


def test_interpret_compression_response_omits_range() -> None:
    """Code mode should omit a contiguous range of lines."""
    result = interpret_compression_response(
        '{"type":"code","content":["2-4:middle content"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.effective_output == "\n".join(
        [
            "alpha",
            "(compressed 3 lines: middle content)",
            "epsilon",
        ]
    )


def test_interpret_compression_response_preserves_content_in_normalization() -> None:
    """Normalized completion should preserve the content array as-is."""
    result = interpret_compression_response(
        '{"type":"code","content":["1-2:header stuff","5:footer"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.normalized_completion == (
        '{"type":"code","content":["1-2:header stuff","5:footer"]}'
    )


def test_interpret_compression_response_omit_without_summary() -> None:
    """Omit entry without colon should use default marker."""
    result = interpret_compression_response(
        '{"type":"code","content":["2"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.valid_response is True
    assert result.effective_output == "\n".join(
        [
            "alpha",
            "(compressed 1 lines)",
            "gamma",
            "delta",
            "epsilon",
        ]
    )


def test_interpret_compression_response_falls_back_for_overlapping_ranges() -> None:
    """Overlapping omit ranges should fall back to the original output."""
    result = interpret_compression_response(
        '{"type":"code","content":["2-4:a","3-5:b"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is True
    assert result.valid_response is False
    assert result.effective_output == "raw original output"


def test_interpret_compression_response_falls_back_for_out_of_range() -> None:
    """Out-of-range omit references should fall back to the original output."""
    result = interpret_compression_response(
        '{"type":"code","content":["20:out of range"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is True
    assert result.valid_response is False
    assert result.effective_output == "raw original output"


def test_interpret_compression_response_requires_list_content_for_code() -> None:
    """Code responses with non-list content should fall back to the original output."""
    result = interpret_compression_response(
        '{"type":"code","content":"summary"}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is True
    assert result.valid_response is False
    assert result.effective_output == "raw original output"


def test_interpret_compression_response_omit_all_lines() -> None:
    """Omitting all lines should produce only markers."""
    result = interpret_compression_response(
        '{"type":"code","content":["1-5:entire output"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.valid_response is True
    assert result.effective_output == "(compressed 5 lines: entire output)"


def test_interpret_compression_response_falls_back_for_invalid_range_syntax() -> None:
    """Malformed omit range should fall back to the original output."""
    result = interpret_compression_response(
        '{"type":"code","content":["3-:bad range"]}',
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is True
    assert result.valid_response is False
    assert result.effective_output == "raw original output"


def test_interpret_compression_response_legacy_text_protocol_falls_back() -> None:
    """Legacy text protocol should no longer be accepted."""
    result = interpret_compression_response(
        "FORMAT: PLAIN\nCONTENT:\nshort summary",
        original_output="raw original output",
        numbered_output=_numbered_output(),
    )

    assert result.keep_original is True
    assert result.valid_response is False
    assert result.effective_output == "raw original output"
