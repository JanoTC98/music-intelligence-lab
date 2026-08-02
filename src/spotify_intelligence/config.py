from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from spotify_intelligence.data.contracts import DataContractError


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file from the versioned ``configs/`` tree."""
    config_path = Path(path)
    if not config_path.exists():
        raise DataContractError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def load_recommender_features(
    path: str | Path = "configs/recommender_features.yaml",
) -> dict[str, Any]:
    """Load the recommender feature and retrieval configuration."""
    return load_yaml_config(path)
