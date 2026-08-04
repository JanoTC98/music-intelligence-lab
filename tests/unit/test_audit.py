import json
import tempfile
from pathlib import Path

import pandas as pd

from spotify_intelligence.data.audit import audit_dataset, compute_file_hash


def test_compute_file_hash():
    path = "data/raw/dataset.csv"
    h = compute_file_hash(path)
    assert isinstance(h, str)
    assert len(h) == 64


def test_raw_dataset_hash_regression():
    """The raw dataset is immutable (sección 12); its SHA-256 is a regression anchor."""
    expected = "b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd"
    assert compute_file_hash("data/raw/dataset.csv") == expected


def test_audit_dataset_returns_report():
    df = pd.read_csv("data/raw/dataset.csv")
    report = audit_dataset(df, "data/raw/dataset.csv")
    assert isinstance(report, dict)
    assert report["raw_rows"] == 114000
    assert "dataset_hash" in report
    assert "nulls" in report
    assert "genres" in report


def test_audit_dataset_writes_json():
    df = pd.read_csv("data/raw/dataset.csv")
    with tempfile.TemporaryDirectory() as tmp:
        audit_dataset(df, "data/raw/dataset.csv", output_dir=tmp)
        json_path = Path(tmp) / "data_quality_report.json"
        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["raw_rows"] == 114000


def test_audit_numbers_match_agents():
    df = pd.read_csv("data/raw/dataset.csv")
    report = audit_dataset(df, "data/raw/dataset.csv")

    assert report["genres"]["total_unique"] == 114
    assert report["track_ids_unique"] == 89741
    assert report["rows_minus_unique_ids"] == 114000 - 89741
    assert report["duration"]["short_tracks_under_60s"] == 851
    assert report["duration"]["long_tracks_over_10min"] == 603
    assert report["popularity"]["zero_count"] == 16020
    assert report["tempo_zero_count"] == 157
    assert report["extremes"]["loudness"]["min"] < report["extremes"]["loudness"]["max"]
