"""Attribute extraction placeholder."""

from __future__ import annotations

from pathlib import Path


class AttributeExtractor:
    """Extracts structured fashion attributes from images or text."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    def extract(self, image_path: Path) -> dict[str, str]:
        """Extract structured attributes for a catalog image."""
        raise NotImplementedError
