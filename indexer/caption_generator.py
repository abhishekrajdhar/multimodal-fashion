"""Caption generation placeholder."""

from __future__ import annotations

from pathlib import Path


class CaptionGenerator:
    """Generates natural-language captions for fashion assets."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def generate(self, image_path: Path) -> str:
        """Generate a caption for a product or scene image."""
        raise NotImplementedError
