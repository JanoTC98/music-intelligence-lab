"""Validate the processed data artifacts (AGENTS.md sección 25.1, sección 9).

Checks the contracts of the generated Parquet tables without modifying them.
Exits with a non-zero status and a clear message on the first failure.

Usage:
    uv run python scripts/validate_processed_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROCESSED_DIR = Path("data/processed")
QUARANTINE_DIR = Path("data/quarantine")
RAW_DATASET = Path("data/raw/dataset.csv")
CONFIG_PATH = Path("configs/data_rules.yaml")

TRACKS_REQUIRED_COLUMNS = [
    "track_id",
    "track_name",
    "track_name_normalized",
    "artists",
    "artists_normalized",
    "album_name",
    "popularity_min",
    "popularity_max",
    "popularity_median",
    "popularity_observations",
    "duration_ms",
    "duration_min",
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
    "recording_group_id",
]

EXPECTED_GENRES = 114


def _load_regression_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    regression = config.get("regression", {})
    if not regression:
        _fail("No existe el bloque regression en configs/data_rules.yaml")
    return regression


def _fail(message: str) -> None:
    print(f"FALLO: {message}")
    sys.exit(1)


def _compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        _fail(f"No existe {path}")
    return pd.read_parquet(path)


def main() -> None:
    if not RAW_DATASET.exists():
        _fail(f"No existe el dataset bruto {RAW_DATASET}")

    regression = _load_regression_config()
    expected_sha256 = regression["raw_dataset_sha256"]
    expected_recording_groups = regression["exact_recording_groups_after_quarantine"]

    tracks = _read("tracks.parquet")
    recordings = _read("recordings.parquet")
    recording_tracks = _read("recording_tracks.parquet")
    genre_catalog = _read("genre_catalog.parquet")

    if not tracks["track_id"].is_unique:
        _fail("tracks.track_id no es �nico")
    if not recordings["recording_group_id"].is_unique:
        _fail("recordings.recording_group_id no es �nico")
    if tracks["recording_group_id"].isna().any():
        _fail("tracks.recording_group_id tiene nulos")
    if len(recordings) != expected_recording_groups:
        _fail(
            f"recordings tiene {len(recordings)} filas; "
            f"esperado {expected_recording_groups} (regresi�n)"
        )

    missing = set(TRACKS_REQUIRED_COLUMNS) - set(tracks.columns)
    if missing:
        _fail(f"tracks.parquet no tiene columnas requeridas: {sorted(missing)}")

    orphans = set(tracks["track_id"]) - set(recording_tracks["track_id"])
    if orphans:
        _fail(f"track_ids sin grupo en recording_tracks: {len(orphans)}")

    if len(genre_catalog) != EXPECTED_GENRES:
        _fail(f"genre_catalog tiene {len(genre_catalog)} g�neros; esperado {EXPECTED_GENRES}")

    track_genres = _read("track_genres.parquet")
    duplicated = track_genres.duplicated(subset=["track_id", "track_genre"]).sum()
    if duplicated:
        _fail(f"track_genres tiene {duplicated} duplicados track_id-g�nero")

    if not (QUARANTINE_DIR / "invalid_identity.parquet").exists():
        _fail("No existe data/quarantine/invalid_identity.parquet")

    manifest_path = PROCESSED_DIR / "prepare_data_manifest.json"
    if not manifest_path.exists():
        _fail("No existe data/processed/prepare_data_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("dataset_sha256") != expected_sha256:
        _fail("El hash del dataset bruto difiere del manifiesto")

    print(
        f"OK: tracks={len(tracks)}, recordings={len(recordings)}, generos={len(genre_catalog)}, raw hash verificado"
    )


if __name__ == "__main__":
    main()
