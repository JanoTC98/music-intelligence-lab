from __future__ import annotations

from pathlib import Path

import pandas as pd

from spotify_intelligence.data.contracts import (
    DataContractError,
    load_rules_config,
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if not path.is_file():
        raise DataContractError(f"Path is not a file: {path}")

    df: pd.DataFrame = pd.read_csv(path)
    return df


def load_dataset_from_config(
    config_path: str | Path = "configs/data_rules.yaml",
    dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    config = load_rules_config(config_path)
    if dataset_path is None:
        dataset_path = config["paths"]["raw_dataset"]
    return load_dataset(dataset_path)
