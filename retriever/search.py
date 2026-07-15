"""Search placeholder."""

from __future__ import annotations

from typing import Any


class SearchEngine:
    """Performs first-stage retrieval over the vector index."""

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return the top matching catalog items."""
        raise NotImplementedError
