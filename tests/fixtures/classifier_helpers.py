"""Tiny synthetic classification fixtures for unit tests.

Produces small ``recordings``, ``recording_genres`` and ``genre_catalog``
DataFrames shaped like the processed tables, plus a frozen grouped split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = [
    "rock",
    "pop",
    "jazz",
    "electronic",
    "classical",
    "hip-hop",
]


def make_genre_catalog() -> pd.DataFrame:
    return pd.DataFrame({"genre_id": range(1, len(GENRES) + 1), "track_genre": GENRES})


def make_recordings(n: int = 24) -> pd.DataFrame:
    """Deterministic recordings table with a few incomplete-audio rows."""
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        rows.append(
            {
                "recording_group_id": f"group_{i:03d}",
                "representative_track_id": f"track_{i:03d}",
                "track_name": f"Song {i}",
                "artists": f"Artist {i % 4}",
                "album_name": f"Album {i}",
                "track_id_count": 1,
                "genre_count": int(rng.integers(1, 4)),
                "artist_count": 1,
                "popularity_median": float(rng.integers(0, 100)),
                "duration_ms": float(rng.integers(90_000, 300_000)),
                "explicit": bool(i % 3),
                "danceability": float(rng.uniform(0, 1)),
                "energy": float(rng.uniform(0, 1)),
                "key": int(rng.integers(0, 12)),
                "loudness": float(rng.uniform(-20, 0)),
                "mode": int(rng.integers(0, 2)),
                "speechiness": float(rng.uniform(0, 1)),
                "acousticness": float(rng.uniform(0, 1)),
                "instrumentalness": float(rng.uniform(0, 1)),
                "liveness": float(rng.uniform(0, 1)),
                "valence": float(rng.uniform(0, 1)),
                "tempo": float(rng.uniform(60, 180)),
                "time_signature": int(rng.choice([3, 4, 4, 4, 5])),
                "audio_analysis_incomplete": i in (2, 9),
            }
        )
    return pd.DataFrame(rows)


def make_recording_genres(recordings: pd.DataFrame) -> pd.DataFrame:
    """Assign one to three genres per group, deterministic by index."""
    rng = np.random.default_rng(5)
    records = []
    for row in recordings.itertuples():
        n_genres = 1 + int(rng.integers(0, 3))
        for genre in rng.choice(GENRES, size=n_genres, replace=False):
            records.append({"recording_group_id": row.recording_group_id, "track_genre": genre})
    return pd.DataFrame(records)


def make_synthetic_split(recordings: pd.DataFrame) -> dict[str, list[str]]:
    """A fixed grouped split used by fixtures (kept deterministic)."""
    groups = recordings["recording_group_id"].tolist()
    train = groups[: int(len(groups) * 0.7)]
    validation = groups[int(len(groups) * 0.7) : int(len(groups) * 0.85)]
    test = groups[int(len(groups) * 0.85) :]
    return {"train": train, "validation": validation, "test": test}


def make_classification_fixture() -> dict:
    """Return recordings, genres, catalog and a split mapping in one dict."""
    recordings = make_recordings(n=24)
    recording_genres = make_recording_genres(recordings)
    genre_catalog = make_genre_catalog()
    split_map = make_synthetic_split(recordings)
    return {
        "recordings": recordings,
        "recording_genres": recording_genres,
        "genre_catalog": genre_catalog,
        "split_map": split_map,
    }
