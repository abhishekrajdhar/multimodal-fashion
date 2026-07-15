"""Image encoder placeholder."""

from __future__ import annotations

from pathlib import Path


class ImageEncoder:
    """Encodes product images into dense embeddings."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def encode(self, image_path: Path) -> list[float]:
        """Generate an embedding for a single product image."""
        raise NotImplementedError
