"""Checkpoint utilities for long-running data processing tasks.

Provides a simple checkpoint mechanism to track completed items
and support resume functionality.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for data processing tasks.

    Tracks completed item IDs and provides save/load functionality.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        self.checkpoint_path = checkpoint_path
        self._completed_ids: set[str] = set()
        self._loaded = False

    def load(self) -> set[str]:
        """Load checkpoint from file.

        Returns:
            Set of completed item IDs.
        """
        if self._loaded:
            return self._completed_ids

        self._loaded = True

        if not self.checkpoint_path.exists():
            return self._completed_ids

        try:
            with open(self.checkpoint_path) as f:
                checkpoint = json.load(f)
            self._completed_ids = set(checkpoint.get("completed_ids", []))
            logger.info(f"Loaded checkpoint: {len(self._completed_ids)} completed items")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            self._completed_ids = set()

        return self._completed_ids

    def save(self, completed_ids: list[str]) -> None:
        """Save checkpoint to file.

        Args:
            completed_ids: List of completed item IDs.
        """
        checkpoint = {
            "completed_ids": completed_ids,
            "timestamp": datetime.now().isoformat(),
        }

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.debug(f"Saved checkpoint: {len(completed_ids)} completed items")

    def is_completed(self, item_id: str) -> bool:
        """Check if an item has been completed.

        Args:
            item_id: The item ID to check.

        Returns:
            True if the item is in the completed set.
        """
        if not self._loaded:
            self.load()
        return item_id in self._completed_ids

    def mark_completed(self, item_id: str) -> None:
        """Mark an item as completed and save checkpoint.

        Args:
            item_id: The item ID to mark as completed.
        """
        if not self._loaded:
            self.load()

        self._completed_ids.add(item_id)
        self.save(list(self._completed_ids))

    def get_completed_count(self) -> int:
        """Get the count of completed items.

        Returns:
            Number of completed items.
        """
        if not self._loaded:
            self.load()
        return len(self._completed_ids)
