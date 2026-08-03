"""Build the versioned preference-recommender artifacts.

Usage:
    uv run python scripts/build_preference_recommender.py

Output (models/preferences/<version>/):
    scaler.joblib, catalog_matrix.npy, catalog_index.parquet,
    ood_reference.json, manifest.json
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from spotify_intelligence.config import load_recommender_features
from spotify_intelligence.data.audit import compute_file_hash
from spotify_intelligence.features.presets import BASIC_FEATURES
from spotify_intelligence.recommenders.errors import IncompatibleArtifactError

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
    *BASIC_FEATURES,
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
    features_config_path: str | Path = "configs/recommender_features.yaml",
    presets_config_path: str | Path = "configs/presets.yaml",
    output_dir: str | Path = "models/preferences",
    version: str = VERSION,
) -> dict:
    """Build the preference recommender artifacts and return the manifest."""
    config = load_recommender_features(features_config_path)
    pref_config = config["preference_recommender"]
    basic_features = tuple(pref_config["basic_features"])
    ood_config = pref_config["out_of_distribution"]

    if list(basic_features) != list(BASIC_FEATURES):
        raise IncompatibleArtifactError(
            f"Configured features {basic_features} differ from supported {BASIC_FEATURES}"
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

    matrix = catalog_index[list(basic_features)].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    centroid = scaled.mean(axis=0)
    distances_to_centroid = np.linalg.norm(scaled - centroid, axis=1)
    p95 = float(np.percentile(distances_to_centroid, ood_config["warning_percentile"]))
    p99 = float(np.percentile(distances_to_centroid, ood_config["weak_match_percentile"]))
    ood_reference = {
        "centroid": centroid.tolist(),
        "percentiles": {"95": p95, "99": p99},
        "warning_percentile": ood_config["warning_percentile"],
        "weak_match_percentile": ood_config["weak_match_percentile"],
    }

    out_path = Path(output_dir) / version
    out_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, out_path / "scaler.joblib")
    np.save(out_path / "catalog_matrix.npy", scaled)
    catalog_index.to_parquet(out_path / "catalog_index.parquet", index=False)
    with open(out_path / "ood_reference.json", "w", encoding="utf-8") as f:
        json.dump(ood_reference, f, indent=2, ensure_ascii=False)

    manifest = {
        "version": version,
        "pipeline": "preference_recommender",
        "features": list(basic_features),
        "scaler": "standard",
        "distance": pref_config["distance"],
        "weight_scale": {"min": 0, "max": 3},
        "out_of_distribution": ood_config,
        "diversity": pref_config["diversity"],
        "filters": pref_config.get("filters", {}),
        "presets_config": presets_config_path,
        "catalog_size": int(len(catalog_index)),
        "dataset_sha256": compute_file_hash("data/raw/dataset.csv"),
        "config_sha256": compute_file_hash(features_config_path),
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
