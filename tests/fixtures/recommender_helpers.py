from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from spotify_intelligence.recommenders.scoring import RECOMMENDER_FEATURES

FEATURES = list(RECOMMENDER_FEATURES)


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
        }
        for feature in FEATURES:
            row[feature] = float(rng.uniform(0.0, 1.0))
        row["tempo"] = 90.0 + i * 8.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_tiny_recommender(artifact_dir: str | Path) -> Path:
    """Create a minimal versioned artifact set and return its directory."""
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = make_catalog_index(n=8)
    matrix = catalog[FEATURES].to_numpy(dtype=float)
    scaled = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

    neighbors = NearestNeighbors(n_neighbors=len(scaled), metric="cosine", algorithm="brute")
    neighbors.fit(scaled)

    scaler = StandardScaler()
    scaler.fit(matrix)

    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(neighbors, out / "neighbors.joblib")
    np.save(out / "catalog_matrix.npy", scaled)
    catalog.to_parquet(out / "catalog_index.parquet", index=False)

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
        "catalog_size": len(catalog),
        "artifact_dir": str(out),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out
