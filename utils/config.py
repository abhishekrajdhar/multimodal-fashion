"""Configuration utilities placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load application configuration from a YAML file."""
    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}

    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a YAML object at the top level.")

    return payload
