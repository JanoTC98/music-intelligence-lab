"""Build the versioned track-recommender artifacts.

Usage:
    uv run python scripts/build_recommender.py

Output (models/recommender/<version>/):
    scaler.joblib, neighbors.joblib, catalog_matrix.npy,
    catalog_index.parquet, manifest.json
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from spotify_intelligence.config import load_recommender_features
from spotify_intelligence.data.audit import compute_file_hash
from spotify_intelligence.recommenders.errors import IncompatibleArtifactError
from spotify_intelligence.recommenders.scoring import RECOMMENDER_FEATURES, create_scaler

VERSION = "v1"
CATALOG_INDEX_COLUMNS = [
    "recording_group_id",
    "representative_track_id",
    "track_name",
    "artists",
    "album_name",
    "genres",
    "duration_ms",
    "popularity_median",
    "explicit",
    *RECOMMENDER_FEATURES,
]


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


def build(
    recordings_path: str | Path = "data/processed/recordings.parquet",
    recording_genres_path: str | Path = "data/processed/recording_genres.parquet",
    config_path: str | Path = "configs/recommender_features.yaml",
    output_dir: str | Path = "models/recommender",
    version: str = VERSION,
) -> dict:
    """Build the recommender artifacts and return the manifest."""
    config = load_recommender_features(config_path)
    track_config = config["track_recommender"]
    features = tuple(track_config["features"])
    baseline = track_config["baseline"]

    if list(features) != list(RECOMMENDER_FEATURES):
        raise IncompatibleArtifactError(
            f"Configured features {features} differ from supported {RECOMMENDER_FEATURES}"
        )

    recordings = pd.read_parquet(recordings_path)
    recording_genres = pd.read_parquet(recording_genres_path)

    eligible = recordings[~recordings["audio_analysis_incomplete"]].reset_index(drop=True)

    genres_by_group = (
        recording_genres.groupby("recording_group_id")["track_genre"].apply(sorted).to_dict()
    )

    catalog_index = eligible[[c for c in CATALOG_INDEX_COLUMNS if c != "genres"]].copy()
    catalog_index["genres"] = catalog_index["recording_group_id"].map(genres_by_group)
    catalog_index = catalog_index.sort_values("recording_group_id").reset_index(drop=True)

    matrix = catalog_index[list(features)].to_numpy(dtype=float)
    scaler = create_scaler(baseline["scaler"])
    scaled = scaler.fit_transform(matrix)

    algorithm = baseline.get("algorithm", "brute")
    neighbors = NearestNeighbors(
        n_neighbors=min(100, len(scaled)),
        metric=baseline["metric"],
        algorithm=algorithm,
    )
    neighbors.fit(scaled)

    out_path = Path(output_dir) / version
    out_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, out_path / "scaler.joblib")
    joblib.dump(neighbors, out_path / "neighbors.joblib")
    np.save(out_path / "catalog_matrix.npy", scaled)
    catalog_index.to_parquet(out_path / "catalog_index.parquet", index=False)

    manifest = {
        "version": version,
        "pipeline": "track_recommender",
        "features": list(features),
        "scaler": baseline["scaler"],
        "metric": baseline["metric"],
        "algorithm": algorithm,
        "retrieval": track_config["retrieval"],
        "exclusions": track_config["exclusions"],
        "catalog_size": int(len(catalog_index)),
        "dataset_sha256": compute_file_hash("data/raw/dataset.csv"),
        "config_sha256": compute_file_hash(config_path),
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifact_dir": str(out_path),
    }
    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    manifest = build()
    print(json.dumps(manifest, indent=2))
