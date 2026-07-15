"""FAISS integration placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class FaissManager:
    """Manages lifecycle operations for the FAISS index."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir

    def create_index(self, embeddings: list[list[float]]) -> None:
        """Create a FAISS index from catalog embeddings."""
        raise NotImplementedError

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Search the FAISS index and return matched items."""
        raise NotImplementedError
