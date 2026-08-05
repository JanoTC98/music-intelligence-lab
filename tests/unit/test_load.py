from pathlib import Path

import pandas as pd
import pytest

from spotify_intelligence.data.contracts import (
    DataContractError,
    get_required_columns,
    load_rules_config,
)
from spotify_intelligence.data.load import load_dataset, load_dataset_from_config

_RAW_DATASET_PRESENT = Path("data/raw/dataset.csv").exists()
_requires_raw_dataset = pytest.mark.skipif(
    not _RAW_DATASET_PRESENT,
    reason="data/raw/dataset.csv not available (raw dataset is not versioned)",
)


@_requires_raw_dataset
def test_load_dataset_returns_dataframe():
    path = "data/raw/dataset.csv"
    df = load_dataset(path)
    assert isinstance(df, pd.DataFrame)


@_requires_raw_dataset
def test_load_dataset_expected_shape():
    path = "data/raw/dataset.csv"
    df = load_dataset(path)
    assert df.shape == (114000, 21)


@_requires_raw_dataset
def test_load_dataset_has_required_columns():
    path = "data/raw/dataset.csv"
    df = load_dataset(path)
    required = get_required_columns()
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent.csv")


@_requires_raw_dataset
def test_load_dataset_via_config():
    df = load_dataset_from_config()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 114000


def test_load_rules_config():
    config = load_rules_config("configs/data_rules.yaml")
    assert "version" in config
    assert "paths" in config
    assert "raw_dataset" in config["paths"]
    assert "validation_ranges" in config


def test_load_rules_config_not_found():
    with pytest.raises(DataContractError):
        load_rules_config("nonexistent.yaml")
