"""Small, dependency-light utilities shared by project modules."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    return data or {}
