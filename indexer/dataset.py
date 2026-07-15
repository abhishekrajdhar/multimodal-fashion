"""Dataset interfaces for the indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class FashionDataset:
    """Placeholder dataset abstraction for fashion retrieval."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def load_records(self) -> list[dict[str, Any]]:
        """Load raw or processed fashion records for indexing."""
        raise NotImplementedError
