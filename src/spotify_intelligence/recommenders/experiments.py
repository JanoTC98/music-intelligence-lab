from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from spotify_intelligence.recommenders.scoring import (
    RECOMMENDER_FEATURES,
    cosine_similarity_from_distance,
    create_scaler,
)

EXP_ID_NAMES = {
    ("standard", "cosine"): "R1",
    ("robust", "cosine"): "R2",
    ("standard", "euclidean"): "R3",
    ("robust", "euclidean"): "R4",
}


def build_experiment_index(
    recordings: pd.DataFrame,
    features: tuple[str, ...] = RECOMMENDER_FEATURES,
    scaler_name: str = "standard",
    metric: str = "cosine",
    algorithm: str = "brute",
) -> tuple[np.ndarray, Any]:
    """Fit scaler + neighbors index for one experiment configuration."""
    matrix = recordings[list(features)].to_numpy(dtype=float)
    scaler = create_scaler(scaler_name)
    scaled = scaler.fit_transform(matrix)
    neighbors = NearestNeighbors(
        n_neighbors=min(100, len(scaled)),
        metric=metric,
        algorithm=algorithm,
    )
    neighbors.fit(scaled)
    return scaled, neighbors


def recommend_with_index(
    scaled_matrix: np.ndarray,
    neighbors: Any,
    seed_row: int,
    top_n: int = 10,
    candidate_floor: int = 100,
) -> tuple[list[int], list[float]]:
    """Retrieve Top-N candidates excluding the seed row."""
    query = scaled_matrix[seed_row : seed_row + 1]
    n_neighbors = min(candidate_floor + 1, scaled_matrix.shape[0])
    distances, indices = neighbors.kneighbors(query, n_neighbors=n_neighbors)
    rows: list[int] = []
    dists: list[float] = []
    for row_idx, distance in zip(indices[0], distances[0], strict=False):
        if int(row_idx) != seed_row:
            rows.append(int(row_idx))
            dists.append(float(distance))
    return rows[:top_n], dists[:top_n]


def run_experiment(
    recordings: pd.DataFrame,
    seed_rows: list[int],
    *,
    scaler_name: str,
    metric: str,
    top_n: int = 10,
    candidate_floor: int = 100,
) -> dict[str, Any]:
    """Run one configuration (R1..R4) over a set of manual seeds."""
    scaled, neighbors = build_experiment_index(recordings, scaler_name=scaler_name, metric=metric)

    start = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    self_recommendations = 0
    duplicates = 0
    similarities: list[float] = []
    unique_groups: set[str] = set()

    for seed_row in seed_rows:
        rows, dists = recommend_with_index(
            scaled, neighbors, seed_row, top_n=top_n, candidate_floor=candidate_floor
        )
        group_ids = recordings.iloc[rows]["recording_group_id"].tolist()
        seed_group = recordings.iloc[seed_row]["recording_group_id"]
        if seed_group in group_ids:
            self_recommendations += 1
        duplicates += int(len(group_ids) - len(set(group_ids)))
        unique_groups.update(group_ids)
        similarities.extend(cosine_similarity_from_distance(d) for d in dists)
        per_seed.append({"seed_row": seed_row, "n": len(rows)})

    elapsed = time.perf_counter() - start

    n_queries = len(seed_rows)
    coverage = len(unique_groups) / len(recordings) if len(recordings) else 0.0
    return {
        "id": EXP_ID_NAMES[(scaler_name, metric)],
        "scaler": scaler_name,
        "metric": metric,
        "comparable_similarity": metric == "cosine",
        "queries": n_queries,
        "total_results": sum(s["n"] for s in per_seed),
        "self_recommendations": self_recommendations,
        "duplicate_groups": duplicates,
        "mean_similarity": float(np.mean(similarities)) if similarities else None,
        "catalog_coverage": float(coverage),
        "latency_total_s": round(elapsed, 4),
        "latency_mean_ms": round(elapsed / n_queries * 1000, 3) if n_queries else None,
    }


def run_all_experiments(
    recordings: pd.DataFrame,
    seed_rows: list[int],
    *,
    top_n: int = 10,
    candidate_floor: int = 100,
) -> pd.DataFrame:
    """Run the four mandatory configurations and return a comparison table."""
    experiments = [
        {"scaler_name": "standard", "metric": "cosine"},
        {"scaler_name": "robust", "metric": "cosine"},
        {"scaler_name": "standard", "metric": "euclidean"},
        {"scaler_name": "robust", "metric": "euclidean"},
    ]
    results = [
        run_experiment(
            recordings,
            seed_rows,
            top_n=top_n,
            candidate_floor=candidate_floor,
            **exp,
        )
        for exp in experiments
    ]
    return pd.DataFrame(results)


def save_experiment_report(
    summary: pd.DataFrame,
    per_seed: pd.DataFrame | None = None,
    output_dir: str | Path = "reports/experiments",
) -> Path:
    """Write the comparison summary and per-seed detail as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "track_recommender_r1_r4_comparison.json"
    summary.to_json(summary_path, orient="records", indent=2)
    return summary_path
