from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spotify_intelligence.recommenders.track_based import (
    RecommendationFilters,
    TrackRecommender,
)


def evaluate_track_recommender(
    recommender: TrackRecommender,
    seed_rows: list[int],
    *,
    top_n: int = 10,
    filters: RecommendationFilters | None = None,
    perturbation_std: float = 0.01,
    n_stability_samples: int = 20,
) -> dict[str, Any]:
    """Evaluate the mandatory sección 14.11 metrics over a sample of seeds.

    Uses the catalog already embedded in the recommender artifacts, so the
    metric values reflect the production index.
    """
    filters = filters or RecommendationFilters()
    catalog = recommender.catalog_index

    self_recommendations = 0
    duplicate_groups = 0
    similarity_values: list[float] = []
    latency_values: list[float] = []
    unique_artists_per_list: list[float] = []
    internal_diversity: list[float] = []
    covered_groups: set[str] = set()
    filter_violations = 0
    total_results = 0

    for row in seed_rows:
        group_id = str(catalog.iloc[row]["recording_group_id"])

        start = time.perf_counter()
        results = recommender.recommend(group_id, top_n=top_n, filters=filters)
        latency_values.append(time.perf_counter() - start)

        total_results += len(results)
        if group_id in set(results["recording_group_id"]):
            self_recommendations += 1
        duplicate_groups += int(
            len(results["recording_group_id"]) - results["recording_group_id"].nunique()
        )

        if "similarity" in results.columns:
            similarity_values.extend(results["similarity"].tolist())
        if len(results) > 1:
            unique_artists_per_list.append(results["artists"].nunique() / len(results))
            internal_diversity.append(results["similarity"].std())
        covered_groups.update(results["recording_group_id"].tolist())

        filter_violations += _count_filter_violations(results, catalog, group_id, filters)

    n_queries = len(seed_rows)
    mean_similarity = float(np.mean(similarity_values)) if similarity_values else None
    latency_ms = [v * 1000 for v in latency_values]

    return {
        "top_n": top_n,
        "queries": n_queries,
        "total_results": total_results,
        "self_recommendations": self_recommendations,
        "duplicate_groups": duplicate_groups,
        "filter_violations": filter_violations,
        "filter_compliance_pct": (
            round(100.0 * (1 - filter_violations / max(total_results, 1)), 2)
            if total_results
            else None
        ),
        "mean_similarity": mean_similarity,
        "catalog_coverage_pct": round(100 * len(covered_groups) / len(catalog), 4),
        "unique_artists_per_list_mean": (
            float(np.mean(unique_artists_per_list)) if unique_artists_per_list else None
        ),
        "internal_diversity_similarity_std": (
            float(np.mean(internal_diversity)) if internal_diversity else None
        ),
        "latency_ms_p50": float(np.percentile(latency_ms, 50)) if latency_ms else None,
        "latency_ms_p95": float(np.percentile(latency_ms, 95)) if latency_ms else None,
        "stability_perturbation_std": perturbation_std,
        "stability_p50_jaccard": _stability_jaccard(
            recommender, seed_rows, top_n, perturbation_std, n_stability_samples
        ),
    }


def _count_filter_violations(
    results: pd.DataFrame,
    catalog: pd.DataFrame,
    seed_group_id: str,
    filters: RecommendationFilters,
) -> int:
    if len(results) == 0:
        return 0
    violations = 0
    for _, row in results.iterrows():
        explicit_violation = (filters.explicit == "explicit" and not bool(row["explicit"])) or (
            filters.explicit == "non_explicit" and bool(row["explicit"])
        )
        if explicit_violation:
            violations += 1
        if filters.genres:
            allowed = set(filters.genres)
            if not (set(row["genres"]) & allowed):
                violations += 1
        if filters.duration_enabled:
            min_ms = filters.duration_min_seconds * 1000
            max_ms = filters.duration_max_seconds * 1000
            if not (min_ms <= int(row["duration_ms"]) <= max_ms):
                violations += 1
        if (
            filters.popularity_min is not None
            and float(row["popularity_median"]) < filters.popularity_min
        ):
            violations += 1
        if filters.different_artist:
            seed_artists = str(
                catalog.iloc[catalog.index[catalog["recording_group_id"] == seed_group_id][0]][
                    "artists"
                ]
            )
            if str(row["artists"]) == seed_artists:
                violations += 1
    return violations


def _stability_jaccard(
    recommender: TrackRecommender,
    seed_rows: list[int],
    top_n: int,
    perturbation_std: float,
    n_samples: int,
) -> float | None:
    """Mean pairwise Jaccard similarity between clean and perturbed results.

    Perturbations are applied to the raw feature matrix of the catalog before
    scaling, so stability reflects robustness to small acoustic changes.
    """
    if perturbation_std <= 0:
        return None

    base_matrix = recommender.catalog_matrix.copy()
    rng = np.random.default_rng(42)
    base_recs = {
        str(recommender.catalog_index.iloc[r]["recording_group_id"]): set(
            recommender.recommend(
                str(recommender.catalog_index.iloc[r]["recording_group_id"]),
                top_n=top_n,
            )["recording_group_id"]
        )
        for r in seed_rows
    }

    jaccards: list[float] = []
    for _ in range(n_samples):
        noise = rng.normal(0, perturbation_std, base_matrix.shape)
        noisy_matrix = base_matrix + noise
        for r in seed_rows:
            group_id = str(recommender.catalog_index.iloc[r]["recording_group_id"])
            perturbed = _recommend_with_matrix(recommender, group_id, top_n, noisy_matrix)
            base_set = base_recs[group_id]
            if not base_set or not perturbed:
                continue
            jaccards.append(len(base_set & perturbed) / len(base_set | perturbed))

    return float(np.mean(jaccards)) if jaccards else None


def _recommend_with_matrix(
    recommender: TrackRecommender,
    seed_group_id: str,
    top_n: int,
    noisy_matrix: np.ndarray,
) -> set[str]:
    """Rank candidates using a noisy matrix but the production neighbor index.

    Used only for stability estimation; does not modify stored artifacts.
    """
    row = recommender._group_to_row[seed_group_id]
    query = noisy_matrix[row : row + 1]
    k = min(100, noisy_matrix.shape[0])
    distances, indices = recommender.neighbors.kneighbors(query, n_neighbors=k)
    rows = [int(i) for i in indices[0] if int(i) != row][:top_n]
    groups = recommender.catalog_index.iloc[rows]["recording_group_id"].tolist()
    return set(groups)


def save_evaluation_report(
    report: dict[str, Any],
    output_dir: str | Path = "reports/metrics",
) -> Path:
    """Write the evaluation report as JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "track_recommender_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    flat = pd.DataFrame([report])
    csv_path = output_dir / "track_recommender_evaluation.csv"
    flat.to_csv(csv_path, index=False)
    return json_path
