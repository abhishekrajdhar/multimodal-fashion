"""Reranking placeholder."""

from __future__ import annotations

from typing import Any


class Reranker:
    """Refines initial retrieval results using richer signals."""

    def rerank(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reorder first-stage retrieval candidates."""
        raise NotImplementedError
