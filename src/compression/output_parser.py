"""Helpers for resolving JSON compression model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal

from json_repair import repair_json
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationError,
    field_validator,
)

_NUMBERED_LINE_PATTERN = re.compile(r"^(\d+)> (.*)$")
_LINE_REFERENCE_PATTERN = re.compile(r"^(?P<start>[1-9]\d*)(?:-(?P<end>[1-9]\d*))?$")


class _CompressionResponseBase(BaseModel):
    """Base schema for JSON compression responses."""

    model_config = ConfigDict(extra="forbid")


class UnchangedCompressionResponse(_CompressionResponseBase):
    """Schema for unchanged tool-output responses."""

    type: Literal["unchanged"]
    content: None = None


class PlainCompressionResponse(_CompressionResponseBase):
    """Schema for plain-text compression responses."""

    type: Literal["plain"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        """Reject empty plain-text content."""
        if not value.strip():
            raise ValueError("plain compression content cannot be empty")
        return value


class CodeCompressionResponse(_CompressionResponseBase):
    """Schema for code compression responses listing ranges to omit."""

    type: Literal["code"]
    content: list[str] = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, values: list[str]) -> list[str]:
        """Normalize whitespace around omit entries."""
        cleaned_values = [value.strip() for value in values]
        if any(not value for value in cleaned_values):
            raise ValueError("code omit entries cannot be empty")
        return cleaned_values


CompressionResponse = Annotated[
    UnchangedCompressionResponse | PlainCompressionResponse | CodeCompressionResponse,
    Field(discriminator="type"),
]


class CompressionResponseRoot(RootModel[CompressionResponse]):
    """Root schema wrapper for schema-guided JSON repair."""


_COMPRESSION_RESPONSE_ADAPTER = TypeAdapter(CompressionResponse)


@dataclass(frozen=True)
class CompressionOutputResolution:
    """Resolved compression output after applying JSON protocol rules.

    Attributes:
        raw_output: Raw model response.
        effective_output: Text the agent should actually see.
        keep_original: Whether runtime should continue from the original output.
        normalized_completion: Canonical JSON response for valid model outputs.
        used_line_references: Whether the response resolved through code line references.
        valid_response: Whether the model output matched the JSON protocol.
    """

    raw_output: str
    effective_output: str
    normalized_completion: str | None
    keep_original: bool
    used_line_references: bool
    valid_response: bool


def _fallback_to_original(
    raw_output: str,
    original_output: str,
) -> CompressionOutputResolution:
    """Build a fallback resolution that restores the original tool output.

    Args:
        raw_output: Raw model response.
        original_output: Original uncompressed tool output.

    Returns:
        Resolution that restores the original tool output.
    """
    return CompressionOutputResolution(
        raw_output=raw_output,
        effective_output=original_output,
        keep_original=True,
        normalized_completion=None,
        used_line_references=False,
        valid_response=False,
    )


def _parse_numbered_output(numbered_output: str) -> dict[int, str]:
    """Parse numbered prompt-visible output lines.

    Args:
        numbered_output: Prompt-visible output with ``N> `` prefixes.

    Returns:
        Mapping from line number to line content without the numeric prefix.

    Raises:
        ValueError: If any line does not match the numbered format or the input is empty.
    """
    parsed_lines: dict[int, str] = {}
    for raw_line in numbered_output.splitlines():
        match = _NUMBERED_LINE_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"Invalid numbered output line: {raw_line!r}")
        line_number = int(match.group(1))
        parsed_lines[line_number] = match.group(2)

    if not parsed_lines:
        raise ValueError("Numbered output cannot be empty")

    return parsed_lines


@dataclass(frozen=True)
class _OmitEntry:
    """A parsed omit entry with line range and summary."""

    start: int
    end: int
    summary: str | None


def _parse_omit_entries(
    content: list[str],
    *,
    max_line_number: int,
) -> list[_OmitEntry]:
    """Parse and validate code-mode omit entries.

    Each entry has format ``"N:summary"`` or ``"N-M:summary"``. Entries without
    a colon are treated as omit ranges with no summary.

    Args:
        content: Raw content strings from the model response.
        max_line_number: Largest line number in the numbered output.

    Returns:
        Sorted, validated omit entries.

    Raises:
        ValueError: If any entry is malformed, out of range, or overlaps.
    """
    entries: list[_OmitEntry] = []

    for item in content:
        colon_pos = item.find(":")
        if colon_pos >= 0:
            range_str = item[:colon_pos].strip()
            summary = item[colon_pos + 1 :].strip() or None
        else:
            range_str = item.strip()
            summary = None

        match = _LINE_REFERENCE_PATTERN.fullmatch(range_str)
        if match is None:
            raise ValueError(f"Invalid omit range: {range_str!r}")

        start = int(match.group("start"))
        end_group = match.group("end")
        end = int(end_group) if end_group is not None else start

        if start > end:
            raise ValueError(f"Descending range: {range_str!r}")
        if end > max_line_number:
            raise ValueError(f"Range {range_str!r} exceeds max line {max_line_number}")

        entries.append(_OmitEntry(start=start, end=end, summary=summary))

    entries.sort(key=lambda e: e.start)

    for i in range(1, len(entries)):
        if entries[i].start <= entries[i - 1].end:
            raise ValueError(
                f"Overlapping ranges: {entries[i - 1].start}-{entries[i - 1].end} "
                f"and {entries[i].start}-{entries[i].end}"
            )

    return entries


def _reconstruct_output_from_omit_ranges(
    parsed_lines: dict[int, str],
    omit_entries: list[_OmitEntry],
) -> str:
    """Reconstruct output by replacing omitted ranges with summary markers.

    Args:
        parsed_lines: Parsed numbered output lines.
        omit_entries: Sorted, validated omit entries.

    Returns:
        Reconstructed output with omitted ranges replaced by markers.
    """
    max_line = max(parsed_lines)
    result_lines: list[str] = []
    current_line = 1
    entry_idx = 0

    while current_line <= max_line:
        if entry_idx < len(omit_entries) and current_line == omit_entries[entry_idx].start:
            entry = omit_entries[entry_idx]
            line_count = entry.end - entry.start + 1
            if entry.summary:
                result_lines.append(f"(compressed {line_count} lines: {entry.summary})")
            else:
                result_lines.append(f"(compressed {line_count} lines)")
            current_line = entry.end + 1
            entry_idx += 1
        else:
            result_lines.append(parsed_lines[current_line])
            current_line += 1

    return "\n".join(result_lines)


def _parse_compression_response(raw_output: str) -> CompressionResponse | None:
    """Repair and validate a raw JSON compression response.

    Args:
        raw_output: Raw model response text.

    Returns:
        Validated compression response, or ``None`` if repair or validation fails.
    """
    try:
        repaired_object = repair_json(
            raw_output,
            schema=CompressionResponseRoot,
            return_objects=True,
        )
    except Exception:
        return None

    try:
        return _COMPRESSION_RESPONSE_ADAPTER.validate_python(repaired_object)
    except ValidationError:
        return None


def _serialize_response(response: CompressionResponse) -> str:
    """Serialize a validated response into canonical JSON.

    Args:
        response: Validated compression response.

    Returns:
        Canonical JSON string with stable field ordering.
    """
    if isinstance(response, UnchangedCompressionResponse):
        payload = {"type": "unchanged", "content": None}
    elif isinstance(response, PlainCompressionResponse):
        payload = {"type": "plain", "content": response.content}
    else:
        payload = {"type": "code", "content": response.content}

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def interpret_compression_response(
    raw_output: str,
    *,
    original_output: str,
    numbered_output: str,
) -> CompressionOutputResolution:
    """Resolve a JSON compression response into the agent-visible output.

    Args:
        raw_output: Raw model response.
        original_output: Original output to restore on fallback.
        numbered_output: Prompt-visible output with numbered lines.

    Returns:
        Parsed compression-output resolution.
    """
    response = _parse_compression_response(raw_output)
    if response is None:
        return _fallback_to_original(raw_output, original_output)

    if isinstance(response, UnchangedCompressionResponse):
        return CompressionOutputResolution(
            raw_output=raw_output,
            effective_output=original_output,
            keep_original=True,
            normalized_completion=_serialize_response(response),
            used_line_references=False,
            valid_response=True,
        )

    if isinstance(response, PlainCompressionResponse):
        return CompressionOutputResolution(
            raw_output=raw_output,
            effective_output=response.content,
            keep_original=False,
            normalized_completion=_serialize_response(response),
            used_line_references=False,
            valid_response=True,
        )

    try:
        parsed_lines = _parse_numbered_output(numbered_output)
        omit_entries = _parse_omit_entries(
            response.content,
            max_line_number=max(parsed_lines),
        )
        reconstructed_output = _reconstruct_output_from_omit_ranges(
            parsed_lines,
            omit_entries,
        )
    except ValueError:
        return _fallback_to_original(raw_output, original_output)

    return CompressionOutputResolution(
        raw_output=raw_output,
        effective_output=reconstructed_output,
        keep_original=False,
        normalized_completion=_serialize_response(response),
        used_line_references=True,
        valid_response=True,
    )
