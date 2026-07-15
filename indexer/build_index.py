"""Index building workflow placeholder."""

from __future__ import annotations

from pathlib import Path


class IndexBuilder:
    """Coordinates dataset preparation and vector index construction."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def build(self) -> None:
        """Build the multimodal retrieval index."""
        raise NotImplementedError
