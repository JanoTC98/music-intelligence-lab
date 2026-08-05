from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from spotify_intelligence.data.contracts import load_rules_config


def _cleaning_rules(config: dict[str, Any]) -> dict[str, Any]:
    return config["cleaning"]  # type: ignore[no-any-return]


def _incomplete_audio_rules(config: dict[str, Any]) -> dict[str, Any]:
    return config["incomplete_audio"]  # type: ignore[no-any-return]


def clean_records(
    df: pd.DataFrame,
    config_path: str | Path = "configs/data_rules.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid_records, invalid_identity_records).

    Valid records keep all original columns except the artificial index.
    Invalid records (missing identity fields) are isolated for quarantine.
    """
    config = load_rules_config(config_path)
    rules = _cleaning_rules(config)

    work = df.copy()
    drop_columns = [col for col in rules.get("drop_columns", []) if col in work.columns]
    if drop_columns:
        work = work.drop(columns=drop_columns)

    identity_required = rules.get("identity_required", [])
    identity_missing = work[identity_required].isnull().any(axis=1)
    invalid = work[identity_missing].copy()
    valid = work[~identity_missing].copy()
    return valid, invalid


def add_anomaly_flags(
    df: pd.DataFrame,
    config_path: str | Path = "configs/data_rules.yaml",
) -> pd.DataFrame:
    """Add audio_analysis_incomplete, is_short_track and is_long_track flags."""
    config = load_rules_config(config_path)
    cleaning = _cleaning_rules(config)
    incomplete = _incomplete_audio_rules(config)

    result = df.copy()

    pattern = incomplete.get("required_zero_pattern", {})
    mask = pd.Series(True, index=result.index)
    for column, value in pattern.items():
        if column in result.columns:
            mask = mask & (result[column] == value)
    result["audio_analysis_incomplete"] = mask

    short_ms = cleaning.get("short_track_threshold_ms", 60000)
    long_ms = cleaning.get("long_track_threshold_ms", 600000)
    result["is_short_track"] = result["duration_ms"] < short_ms
    result["is_long_track"] = result["duration_ms"] > long_ms

    return result


def _track_first_columns() -> list[str]:
    return [
        "track_id",
        "track_name",
        "track_name_normalized",
        "artists",
        "artists_normalized",
        "album_name",
        "duration_ms",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
        "audio_analysis_incomplete",
        "is_short_track",
        "is_long_track",
    ]


def build_track_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """One row per track_id with aggregated popularity and first audio values."""
    first_columns = [col for col in _track_first_columns() if col in df.columns]
    first = (
        df.sort_values(["track_id", "popularity"])
        .groupby("track_id", as_index=False)[first_columns]
        .first()
    )

    popularity_agg = (
        df.groupby("track_id")["popularity"]
        .agg(
            popularity_min="min",
            popularity_max="max",
            popularity_median="median",
            popularity_observations="count",
        )
        .reset_index()
    )

    catalog = first.merge(popularity_agg, on="track_id", how="left")
    catalog["duration_min"] = catalog["duration_ms"] / 60000.0
    return catalog


def build_track_genres(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (track_id, track_genre)."""
    result = df[["track_id", "track_genre"]].copy()
    result = (
        result.drop_duplicates().sort_values(["track_id", "track_genre"]).reset_index(drop=True)
    )
    return result


def build_genre_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Catalog of the 114 original genres."""
    genres = sorted(df["track_genre"].dropna().unique())
    catalog = pd.DataFrame({"genre_id": range(1, len(genres) + 1), "track_genre": genres})
    return catalog


def build_track_artists(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (track_id, artist) preserving the original text order."""
    rows: list[dict[str, str]] = []
    for track_id, artists_value in zip(df["track_id"], df["artists"], strict=False):
        if pd.isna(artists_value):
            continue
        for artist in str(artists_value).split(";"):
            artist = artist.strip()
            if artist:
                rows.append({"track_id": track_id, "artist": artist})

    result = pd.DataFrame(rows, columns=["track_id", "artist"])
    result = result.drop_duplicates().sort_values(["track_id", "artist"]).reset_index(drop=True)
    return result
