"""Persistent on-disk cache for HF ``Dataset`` pairs used by trainers.

``Dataset.from_list`` assigns a random fingerprint, so TRL's built-in
tokenization (`datasets.map`) and reference log-prob (`.npz`) caches never hit
between runs. This helper persists the constructed datasets once and rehydrates
them with a stable fingerprint via ``save_to_disk``/``load_from_disk``, which
re-enables those downstream caches automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

CacheBuilder = Callable[[], tuple[Dataset, Dataset | None, dict[str, Any]]]


def compute_cache_key(parts: dict[str, Any]) -> str:
    """Compute a short stable hash from a dict of cache-relevant parts.

    Values are encoded via ``repr`` so ints, floats, strings, and tuples
    round-trip unambiguously. Order-independent: keys are sorted first.

    Args:
        parts: Mapping of name -> value for every input that, if changed,
            should invalidate the cache.

    Returns:
        16-character hex digest.
    """
    h = hashlib.sha256()
    for key in sorted(parts):
        h.update(f"{key}={parts[key]!r}|".encode())
    return h.hexdigest()[:16]


def file_fingerprint(path: str | os.PathLike) -> tuple[int, int]:
    """Return ``(size, mtime_ns)`` of ``path`` for cache-key inclusion."""
    st = Path(path).stat()
    return (st.st_size, st.st_mtime_ns)


def load_or_build(
    cache_dir: Path,
    builder: CacheBuilder,
    rebuild: bool = False,
) -> tuple[Dataset, Dataset | None, dict[str, Any]]:
    """Load cached datasets if present, else build and persist them.

    Args:
        cache_dir: Directory under which ``train/``, ``eval/``, and
            ``meta.json`` are stored. Created if missing.
        builder: Zero-arg callable returning ``(train_ds, eval_ds_or_None,
            meta_dict)``. Called only on cache miss or forced rebuild.
        rebuild: When True, delete any existing cache before building.

    Returns:
        The ``(train_dataset, eval_dataset, meta)`` triple, either loaded or
        freshly built.
    """
    train_path = cache_dir / "train"
    eval_path = cache_dir / "eval"
    meta_path = cache_dir / "meta.json"

    if rebuild and cache_dir.exists():
        logger.info(f"Rebuild requested — removing cache dir {cache_dir}")
        shutil.rmtree(cache_dir)

    if train_path.exists() and meta_path.exists():
        train_dataset = Dataset.load_from_disk(str(train_path))
        eval_dataset = Dataset.load_from_disk(str(eval_path)) if eval_path.exists() else None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        eval_suffix = f" / {len(eval_dataset)} eval" if eval_dataset is not None else ""
        logger.info(
            f"Loaded cached datasets from {cache_dir}: {len(train_dataset)} train{eval_suffix}"
        )
        return train_dataset, eval_dataset, meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, eval_dataset, meta = builder()
    train_dataset.save_to_disk(str(train_path))
    if eval_dataset is not None:
        eval_dataset.save_to_disk(str(eval_path))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Saved datasets to cache dir {cache_dir}")

    # Round-trip through disk so the returned dataset carries the same
    # fingerprint that future ``load_from_disk`` calls will observe. Without
    # this, the very first run's downstream caches (TRL's map + ref-logp
    # .npz) would be keyed on the random in-memory fingerprint and get
    # invalidated on the second run.
    train_dataset = Dataset.load_from_disk(str(train_path))
    if eval_dataset is not None:
        eval_dataset = Dataset.load_from_disk(str(eval_path))
    return train_dataset, eval_dataset, meta
