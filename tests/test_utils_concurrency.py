"""Tests for shared concurrency helpers."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.utils.concurrency import gather_bounded, iter_parallel_completed


def test_gather_bounded_preserves_input_order() -> None:
    """Bounded async helper should preserve input order."""

    async def worker(value: int) -> int:
        await asyncio.sleep(0.03 * (4 - value))
        return value

    results = asyncio.run(gather_bounded([1, 2, 3], worker, max_workers=3))

    assert results == [1, 2, 3]


def test_gather_bounded_enforces_concurrency_limit() -> None:
    """Bounded async helper should never exceed the configured concurrency."""
    active_workers = 0
    peak_workers = 0
    lock = asyncio.Lock()

    async def worker(value: int) -> int:
        nonlocal active_workers, peak_workers
        async with lock:
            active_workers += 1
            peak_workers = max(peak_workers, active_workers)
        await asyncio.sleep(0.02)
        async with lock:
            active_workers -= 1
        return value

    results = asyncio.run(gather_bounded([1, 2, 3, 4], worker, max_workers=2))

    assert results == [1, 2, 3, 4]
    assert peak_workers == 2


def test_gather_bounded_propagates_worker_exceptions() -> None:
    """Bounded async helper should re-raise worker exceptions."""

    async def worker(value: int) -> int:
        if value == 2:
            raise RuntimeError("boom")
        return value

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(gather_bounded([1, 2, 3], worker, max_workers=2))


def test_gather_bounded_accepts_empty_input() -> None:
    """Bounded async helper should return an empty list for empty input."""

    async def worker(value: int) -> int:
        return value

    assert asyncio.run(gather_bounded([], worker, max_workers=2)) == []


def test_gather_bounded_rejects_non_positive_worker_count() -> None:
    """Bounded async helper should fail fast on invalid worker counts."""

    async def worker(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="max_workers must be positive"):
        asyncio.run(gather_bounded([1], worker, max_workers=0))


def test_iter_parallel_completed_yields_results_in_completion_order() -> None:
    """Streaming helper should yield results as workers finish."""

    def worker(value: int) -> int:
        time.sleep(0.03 * (4 - value))
        return value

    results = list(
        iter_parallel_completed(
            [1, 2, 3],
            worker,
            max_workers=3,
        )
    )

    assert results == [3, 2, 1]


def test_iter_parallel_completed_propagates_worker_exceptions() -> None:
    """Streaming helper should re-raise worker exceptions."""

    def worker(value: int) -> int:
        if value == 2:
            raise RuntimeError("boom")
        return value

    with pytest.raises(RuntimeError, match="boom"):
        list(iter_parallel_completed([1, 2, 3], worker, max_workers=2))


def test_iter_parallel_completed_accepts_none_results() -> None:
    """Streaming helper should preserve ``None`` worker results."""
    results = list(
        iter_parallel_completed(
            [1, 2],
            lambda value: None if value == 1 else value,
            max_workers=2,
        )
    )

    assert sorted(results, key=lambda value: value is not None) == [None, 2]


def test_iter_parallel_completed_rejects_non_positive_worker_count() -> None:
    """Streaming helper should fail fast on invalid worker counts."""
    with pytest.raises(ValueError, match="max_workers must be positive"):
        list(iter_parallel_completed([1], lambda value: value, max_workers=0))
