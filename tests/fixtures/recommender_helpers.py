from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from spotify_intelligence.features.presets import BASIC_FEATURES
from spotify_intelligence.recommenders.scoring import RECOMMENDER_FEATURES

FEATURES = list(RECOMMENDER_FEATURES)
PREFERENCE_FEATURES = list(BASIC_FEATURES)


def make_catalog_index(n: int = 8) -> pd.DataFrame:
    """Deterministic small catalog with controllable feature spread."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        row = {
            "recording_group_id": f"g{i:02d}",
            "representative_track_id": f"t{i:02d}",
            "track_name": f"Song {i}",
            "artists": f"Artist {i % 3}",
            "album_name": f"Album {i}",
            "genres": [f"genre{i % 2}"],
            "duration_ms": int(120000 + i * 60000),
            "popularity_median": float(20 + i),
            "explicit": bool(i % 2),
            "audio_analysis_incomplete": i == 7,
        }
        for feature in FEATURES:
            row[feature] = float(rng.uniform(0.0, 1.0))
        row["tempo"] = 90.0 + i * 8.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_tiny_recommender(artifact_dir: str | Path, catalog: pd.DataFrame | None = None) -> Path:
    """Create a minimal versioned artifact set and return its directory.

    ``catalog`` defaults to ``make_catalog_index(n=8)``; callers can pass a
    custom catalog to exercise specific cases such as near-duplicate works.
    """
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    if catalog is None:
        catalog = make_catalog_index(n=8)
    eligible = catalog[~catalog["audio_analysis_incomplete"]].reset_index(drop=True)
    matrix = eligible[FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    neighbors = NearestNeighbors(n_neighbors=len(scaled), metric="cosine", algorithm="brute")
    neighbors.fit(scaled)

    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(neighbors, out / "neighbors.joblib")
    np.save(out / "catalog_matrix.npy", scaled)
    eligible.to_parquet(out / "catalog_index.parquet", index=False)

    manifest = {
        "version": "test",
        "pipeline": "track_recommender",
        "features": FEATURES,
        "scaler": "standard",
        "metric": "cosine",
        "algorithm": "brute",
        "retrieval": {
            "default_top_n": 10,
            "min_top_n": 5,
            "max_top_n": 20,
            "initial_candidate_floor": 100,
            "candidate_multiplier": 10,
            "expansion_steps": [500, 2000],
        },
        "exclusions": {"audio_analysis_incomplete": True, "same_recording_group": True},
        "catalog_size": len(eligible),
        "artifact_dir": str(out),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out


def make_same_work_catalog() -> pd.DataFrame:
    """Catalog where ``g00b`` is a near-duplicate release of ``g00``.

    Both share ``track_name`` and ``artists`` but keep a different
    ``recording_group_id`` (the conservative exact grouping, sección 3.3). The audio
    features are perturbed slightly so the duplicate is acoustically the closest
    neighbor of the original.
    """
    catalog = make_catalog_index(n=8)
    duplicate = catalog.iloc[[0]].copy()
    duplicate["recording_group_id"] = "g00b"
    duplicate["representative_track_id"] = "t00b"
    duplicate["album_name"] = "Album 0 (Reissue)"
    for feature in FEATURES:
        duplicate[feature] = duplicate[feature].astype(float) + 1e-4
    return pd.concat([catalog, duplicate], ignore_index=True)


def make_affinity_catalog() -> pd.DataFrame:
    """Catalog where genre-affinity re-ranking is observable.

    All features except ``energy`` are constant (zero variance, handled by the
    scaler), so energy alone drives the ordering. The nearest neighbor
    (``g01``) is a different genre, while the next-closest candidates (``g02``,
    ``g04``, ``g06``) share the seed genre ``genre0``.
    """
    catalog = make_catalog_index(n=8)
    for feature in FEATURES:
        catalog[feature] = 0.5
    catalog["energy"] = [0.500, 0.501, 0.520, 0.600, 0.520, 0.900, 0.560, 0.700]
    return catalog


def build_tiny_preference_recommender(artifact_dir: str | Path) -> Path:
    """Create a minimal preference recommender artifact set and return its dir."""
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = make_catalog_index(n=8)
    matrix = catalog[PREFERENCE_FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    centroid = scaled.mean(axis=0)
    distances = np.linalg.norm(scaled - centroid, axis=1)
    ood_reference = {
        "centroid": centroid.tolist(),
        "percentiles": {
            "95": float(np.percentile(distances, 95)),
            "99": float(np.percentile(distances, 99)),
        },
        "warning_percentile": 95,
        "weak_match_percentile": 99,
    }

    joblib.dump(scaler, out / "scaler.joblib")
    np.save(out / "catalog_matrix.npy", scaled)
    catalog.to_parquet(out / "catalog_index.parquet", index=False)
    with open(out / "ood_reference.json", "w", encoding="utf-8") as f:
        json.dump(ood_reference, f, indent=2)

    manifest = {
        "version": "test",
        "pipeline": "preference_recommender",
        "features": PREFERENCE_FEATURES,
        "scaler": "standard",
        "distance": "weighted_euclidean",
        "weight_scale": {"min": 0, "max": 3},
        "out_of_distribution": {"warning_percentile": 95, "weak_match_percentile": 99},
        "diversity": {"enabled_default": False, "method": "mmr", "lambda_default": 0.85},
        "catalog_size": len(catalog),
        "artifact_dir": str(out),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out
