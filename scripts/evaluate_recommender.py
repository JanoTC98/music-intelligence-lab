"""Run the offline evaluation of the track recommender (AGENTS.md §14.11).

Usage:
    uv run python scripts/evaluate_recommender.py

Output:
    reports/metrics/track_recommender_evaluation.json
    reports/metrics/track_recommender_evaluation.csv
"""

from __future__ import annotations

import json

import numpy as np

from spotify_intelligence.recommenders.evaluation import (
    evaluate_track_recommender,
    save_evaluation_report,
)
from spotify_intelligence.recommenders.track_based import (
    RecommendationFilters,
    TrackRecommender,
)

EVALUATION_SAMPLE_SIZE = 200
TOP_N = 10


def _sample_seed_rows(recommender: TrackRecommender, size: int) -> list[int]:
    rng = np.random.default_rng(42)
    total = len(recommender.catalog_index)
    size = min(size, total)
    return sorted(rng.choice(total, size=size, replace=False).tolist())


if __name__ == "__main__":
    recommender = TrackRecommender("models/recommender/v1")
    seed_rows = _sample_seed_rows(recommender, EVALUATION_SAMPLE_SIZE)
    print(f"Evaluando con {len(seed_rows)} semillas aleatorias (Top-{TOP_N})...")

    report = evaluate_track_recommender(
        recommender,
        seed_rows,
        top_n=TOP_N,
        filters=RecommendationFilters(),
        perturbation_std=0.01,
        n_stability_samples=20,
    )
    path = save_evaluation_report(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Reporte guardado en {path}")
