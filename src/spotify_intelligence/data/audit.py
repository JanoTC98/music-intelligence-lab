from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spotify_intelligence.data.contracts import (
    EXPECTED_GENRE_COUNT,
    EXPECTED_ROWS_PER_GENRE,
    DataContractError,
)
from spotify_intelligence.data.load import load_dataset
from spotify_intelligence.data.validate import (
    check_required_columns,
    detect_incomplete_audio,
    report_column_extremes,
    validate_column_ranges,
)


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    path = Path(path)
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(
    df: pd.DataFrame,
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    report["dataset_hash"] = compute_file_hash(dataset_path)
    report["pipeline_version"] = "1.0"
    report["generated_at_utc"] = datetime.now(UTC).isoformat()

    report["raw_rows"] = len(df)
    report["raw_columns"] = len(df.columns)

    columns_present = list(df.columns)
    report["columns_present"] = columns_present

    try:
        check_required_columns(df)
        report["required_columns_ok"] = True
    except DataContractError as e:
        report["required_columns_ok"] = False
        report["required_columns_error"] = str(e)

    null_counts = df.isnull().sum()
    report["nulls"] = {
        "total_cells": int(null_counts.sum()),
        "columns_with_nulls": {
            str(col): int(null_counts[col]) for col in null_counts.index if null_counts[col] > 0
        },
    }

    report["duplicates"] = {
        "total_duplicate_rows": int(df.duplicated().sum()),
        "track_id_duplicates": int(
            df["track_id"].duplicated().sum() if "track_id" in df.columns else 0
        ),
    }

    if "track_genre" in df.columns:
        genre_counts = df["track_genre"].value_counts()
        report["genres"] = {
            "total_unique": int(genre_counts.count()),
            "expected_unique": EXPECTED_GENRE_COUNT,
            "rows_per_genre_expected": EXPECTED_ROWS_PER_GENRE,
            "genre_counts_all_equal": bool((genre_counts == EXPECTED_ROWS_PER_GENRE).all()),
        }

    if "duration_ms" in df.columns:
        report["duration"] = {
            "short_tracks_under_60s": int((df["duration_ms"] < 60000).sum()),
            "long_tracks_over_10min": int((df["duration_ms"] > 600000).sum()),
        }

    if "popularity" in df.columns:
        report["popularity"] = {
            "zero_count": int((df["popularity"] == 0).sum()),
        }

    if "tempo" in df.columns:
        report["tempo_zero_count"] = int((df["tempo"] == 0).sum())

    incomplete_mask = detect_incomplete_audio(df)
    report["incomplete_audio"] = {
        "count": int(incomplete_mask.sum()),
        "pattern_used": [
            "tempo=0",
            "danceability=0",
            "speechiness=0",
            "valence=0",
            "time_signature=0",
        ],
    }

    range_violations = validate_column_ranges(df)
    report["range_violations"] = {
        col: [{"row": int(r), "value": v} for r, v in violators]
        for col, violators in range_violations.items()
    }

    report["extremes"] = report_column_extremes(df)

    track_ids = df["track_id"].nunique() if "track_id" in df.columns else 0
    report["track_ids_unique"] = int(track_ids)
    report["rows_minus_unique_ids"] = report["raw_rows"] - track_ids

    multi_genre_tracks = 0
    max_genres = 0
    if "track_id" in df.columns and "track_genre" in df.columns:
        per_track = df.groupby("track_id")["track_genre"].nunique()
        multi_genre_tracks = int((per_track > 1).sum())
        max_genres = int(per_track.max())
    report["multi_genre"] = {
        "tracks_with_multiple_genres": multi_genre_tracks,
        "max_genres_per_track": max_genres,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "data_quality_report.json"

        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (np.bool_,)):
                    return bool(o)
                return super().default(o)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    return report


def run_audit(
    dataset_path: str | Path = "data/raw/dataset.csv",
    output_dir: str | Path = "reports/data_quality",
) -> dict[str, Any]:
    df = load_dataset(dataset_path)
    report = audit_dataset(df, dataset_path, output_dir=output_dir)
    return report
