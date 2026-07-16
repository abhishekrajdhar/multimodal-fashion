"""Logging utilities placeholder."""

from __future__ import annotations

import logging
from typing import Any


def setup_logging(config: dict[str, Any]) -> None:
    """Initialize application logging from configuration."""
    level_name = str(config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = str(config.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    date_format = str(config.get("datefmt", "%Y-%m-%d %H:%M:%S"))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        force=True,
    )
