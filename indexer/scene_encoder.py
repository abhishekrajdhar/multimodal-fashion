"""Scene encoder placeholder."""

from __future__ import annotations

from pathlib import Path


class SceneEncoder:
    """Encodes query or catalog scenes into dense embeddings."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def encode(self, image_path: Path) -> list[float]:
        """Generate an embedding for a fashion scene image."""
        raise NotImplementedError
