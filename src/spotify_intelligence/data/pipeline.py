from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from spotify_intelligence.data.audit import compute_file_hash
from spotify_intelligence.data.clean import (
    add_anomaly_flags,
    build_genre_catalog,
    build_track_artists,
    build_track_catalog,
    build_track_genres,
    clean_records,
)
from spotify_intelligence.data.contracts import load_rules_config
from spotify_intelligence.identity.duplicate_candidates import build_near_duplicate_candidates
from spotify_intelligence.identity.normalize import normalize_identity_fields
from spotify_intelligence.identity.recording_groups import (
    assign_recording_group_ids,
    build_anomalies,
    build_recording_genres,
    build_recording_tracks,
    build_recordings,
)

PIPELINE_VERSION = "2.0"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return None


def _write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def prepare_data(
    dataset_path: str | Path = "data/raw/dataset.csv",
    config_path: str | Path = "configs/data_rules.yaml",
    output_root: str | Path = ".",
    *,
    generate_candidates: bool = True,
) -> dict:
    """Run the full preparation pipeline and return its manifest."""
    config = load_rules_config(config_path)
    output_root = Path(output_root)
    quarantine_dir = output_root / config["paths"]["quarantine_dir"]
    processed_dir = output_root / config["paths"]["processed_dir"]
    report_dir = output_root / "reports" / "identity"

    raw = pd.read_csv(dataset_path)
    valid, invalid = clean_records(raw, config_path=config_path)
    valid = add_anomaly_flags(valid, config_path=config_path)
    valid = normalize_identity_fields(valid, config_path=config_path)

    if not invalid.empty:
        _write_parquet(invalid, quarantine_dir / "invalid_identity.parquet")

    tracks = build_track_catalog(valid)
    tracks = assign_recording_group_ids(tracks)
    _write_parquet(tracks, processed_dir / "tracks.parquet")

    track_genres = build_track_genres(valid)
    _write_parquet(track_genres, processed_dir / "track_genres.parquet")

    genre_catalog = build_genre_catalog(valid)
    _write_parquet(genre_catalog, processed_dir / "genre_catalog.parquet")

    track_artists = build_track_artists(valid)
    _write_parquet(track_artists, processed_dir / "track_artists.parquet")

    recording_tracks = build_recording_tracks(tracks)
    _write_parquet(recording_tracks, processed_dir / "recording_tracks.parquet")

    recording_genres = build_recording_genres(recording_tracks, track_genres)
    _write_parquet(recording_genres, processed_dir / "recording_genres.parquet")

    recordings = build_recordings(tracks, recording_genres, track_artists)
    _write_parquet(recordings, processed_dir / "recordings.parquet")

    anomalies = build_anomalies(tracks)
    _write_parquet(anomalies, processed_dir / "anomalies.parquet")

    if generate_candidates:
        candidates = build_near_duplicate_candidates(recordings)
        report_dir.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(report_dir / "near_duplicate_candidates.csv", index=False)

    manifest = {
        "dataset_sha256": compute_file_hash(dataset_path),
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_rows": int(len(raw)),
        "valid_track_ids": int(tracks["track_id"].nunique()),
        "recording_group_count": int(tracks["recording_group_id"].nunique()),
        "rules_config_sha256": compute_file_hash(config_path),
    }

    with open(processed_dir / "prepare_data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
