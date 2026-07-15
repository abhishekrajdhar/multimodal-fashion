"""End-to-end retrieval workflow placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RetrievalService:
    """Coordinates query parsing, search, and reranking."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def retrieve(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Run the multimodal retrieval pipeline for a query."""
        raise NotImplementedError
