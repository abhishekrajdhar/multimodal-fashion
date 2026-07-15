"""Query parsing placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class QueryParser:
    """Parses multimodal user queries into structured components."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path

    def parse(self, query: dict[str, Any]) -> dict[str, Any]:
        """Normalize text, image, and metadata query inputs."""
        raise NotImplementedError
