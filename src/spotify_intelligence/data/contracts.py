from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DataContractError(Exception):
    pass


_REQUIRED_COLUMNS: list[str] = [
    "Unnamed: 0",
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
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
    "track_genre",
]

_VALIDATION_RANGES: dict[str, dict[str, Any]] = {
    "popularity": {"min": 0, "max": 100},
    "duration_ms": {"min": 1},
    "danceability": {"min": 0, "max": 1},
    "energy": {"min": 0, "max": 1},
    "key": {"min": -1, "max": 11},
    "loudness": {"report_extremes": True},
    "mode": {"allowed_values": [0, 1]},
    "speechiness": {"min": 0, "max": 1},
    "acousticness": {"min": 0, "max": 1},
    "instrumentalness": {"min": 0, "max": 1},
    "liveness": {"min": 0, "max": 1},
    "valence": {"min": 0, "max": 1},
    "tempo": {"min": 0},
    "time_signature": {"min": 0, "max": 5},
}

_INCOMPLETE_AUDIO_PATTERN = {
    "tempo": 0,
    "danceability": 0,
    "speechiness": 0,
    "valence": 0,
    "time_signature": 0,
}

EXPECTED_GENRE_COUNT = 114
EXPECTED_ROWS_PER_GENRE = 1000
EXPECTED_TOTAL_ROWS = 114000


def get_required_columns() -> list[str]:
    return list(_REQUIRED_COLUMNS)


def get_validation_ranges() -> dict[str, dict[str, Any]]:
    return dict(_VALIDATION_RANGES)


def get_incomplete_audio_pattern() -> dict[str, int]:
    return dict(_INCOMPLETE_AUDIO_PATTERN)


def load_rules_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise DataContractError(f"Rules config not found: {path}")
    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config
