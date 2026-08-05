"""Exploratory analysis of track-recommender genre coherence.

Read-only experiment: it never retrains and never writes to ``configs/`` or the
model artifacts. It uses the stored recommender artifacts and measures, for a
sample of seeds, how often the Top-10 shares at least one genre with the seed
(``genre_coherence@10``) under three variants:

1. baseline: current production behavior (equal-weight cosine, R1).
2. weighted: weighted Euclidean over the standardized matrix, emphasizing
   tempo, loudness, energy, valence and danceability.
3. genre_affinity: baseline neighbors (Top-100 by cosine) re-ranked so that
   candidates sharing a genre with the seed come first.

Per seed we also compute ``availability``: whether any of the Top-100 acoustic
neighbors shares a genre with the seed. Seeds whose genre is absent from the
acoustic neighborhood cannot be helped by an affinity re-rank.

Output: ``reports/experiments/recommender_relevance_analysis.json`` and a
console summary, including a per-genre breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spotify_intelligence.recommenders.track_based import TrackRecommender  # noqa: E402

ARTIFACT_DIR = ROOT / "models" / "recommender" / "v1"
OUTPUT = ROOT / "reports" / "experiments" / "recommender_relevance_analysis.json"
DEFAULT_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
NEIGHBOR_FLOOR = 100
TOP_N = 10

EXPERIMENT_WEIGHTS = {
    "danceability": 1.5,
    "energy": 1.5,
    "loudness": 1.5,
    "speechiness": 0.5,
    "acousticness": 0.5,
    "instrumentalness": 0.5,
    "liveness": 0.5,
    "valence": 1.5,
    "tempo": 2.0,
}


def _same_work_mask(
    seed_name: str,
    seed_artists: str,
    names: pd.Series,
    artists: pd.Series,
) -> np.ndarray:
    return names.astype(str).str.strip().str.casefold().eq(
        seed_name.strip().casefold()
    ) & artists.astype(str).str.strip().str.casefold().eq(seed_artists.strip().casefold())


def _to_genre_set(value: object) -> set[str]:
    """Convert a genre cell (list, tuple, np.ndarray or None) to a set."""
    if value is None:
        return set()
    if isinstance(value, np.ndarray):
        return {str(item) for item in value.tolist()}
    if isinstance(value, (list, tuple)):
        return {str(item) for item in value}
    return set()


def _coherence(seed_genres: set[str], candidate_genres: list[set[str]]) -> float:
    if len(candidate_genres) == 0:
        return 0.0
    hits = sum(1 for genres in candidate_genres if bool(genres & seed_genres))
    return hits / len(candidate_genres)


def _weighted_metrics(
    matrix: np.ndarray,
    unit: np.ndarray,
    seed_row: int,
    w: np.ndarray,
    genres: list[set[str]],
    mask: np.ndarray,
    artists: pd.Series,
    top_n: int = TOP_N,
) -> tuple[float, float, float, float]:
    """Weighted distance ranking.

    Returns ``(coherence@top_n, mean_cosine, internal_std, artists_prop)``.
    """
    query = matrix[seed_row]
    diff = matrix - query
    dist = np.sqrt((diff**2 * w).sum(axis=1) / w.sum())
    dist[mask] = np.inf
    order = np.argsort(dist)[: top_n + 1]
    order = order[~mask[order]][:top_n]
    cos = (unit[order] * unit[seed_row]).sum(axis=1)
    return (
        _coherence(genres[seed_row], [genres[i] for i in order]),
        float(cos.mean()),
        float(cos.std()),
        float(artists.iloc[order].nunique() / len(order)),
    )


def _affinity_metrics(
    unit: np.ndarray,
    seed_row: int,
    genres: list[set[str]],
    mask: np.ndarray,
    artists: pd.Series,
    top_n: int = TOP_N,
    neighbor_floor: int = NEIGHBOR_FLOOR,
) -> tuple[float, float, float, float, float]:
    """Affinity re-rank over the acoustic neighborhood.

    Returns ``(coherence@top_n, mean_cosine, internal_std, artists_prop,
    availability)``. Availability is the fraction of neighbors sharing a genre
    with the seed; when it is 0.0 the re-rank cannot improve genre coherence.
    """
    query = unit[seed_row]
    cos = (unit * query).sum(axis=1)
    cos[mask] = -np.inf
    neighbors = np.argsort(cos)[::-1][:neighbor_floor]
    affinity = np.array([bool(genres[i] & genres[seed_row]) for i in neighbors], dtype=float)
    order = neighbors[np.lexsort((cos[neighbors], affinity))[::-1]][:top_n]
    return (
        _coherence(genres[seed_row], [genres[i] for i in order]),
        float(cos[order].mean()),
        float(cos[order].std()),
        float(artists.iloc[order].nunique() / len(order)),
        float(affinity.mean()),
    )


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "share_ge_0_5": float((array >= 0.5).mean()),
    }


def _variant_report(
    coh: list[float],
    cos: list[float],
    cos_std: list[float],
    artists: list[float],
    availability: list[float] | None = None,
) -> dict[str, object]:
    """Metrics block for one ranking variant.

    ``internal_diversity_similarity_std`` and ``unique_artists_per_list`` follow
    the official evaluation methodology (mean of per-list std and of per-list
    unique-artist proportion).
    """
    block: dict[str, object] = {
        "genre_coherence_at_10": _distribution(coh),
        "mean_cosine_similarity": float(np.mean(cos)),
        "internal_diversity_similarity_std": _distribution(cos_std),
        "unique_artists_per_list": _distribution(artists),
    }
    if availability is not None:
        block["availability_in_top100"] = float(np.mean(availability))
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of seeds to sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for the sample (default: {RANDOM_STATE}).",
    )
    args = parser.parse_args()

    rec = TrackRecommender(ARTIFACT_DIR)
    matrix = rec.catalog_matrix.astype(float)
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    catalog = rec.catalog_index.reset_index(drop=True)
    names = catalog["track_name"]
    artists = catalog["artists"]
    genre_rows = [_to_genre_set(value) for value in catalog["genres"]]

    features = rec.feature_columns
    w = np.array([EXPERIMENT_WEIGHTS.get(feature, 1.0) for feature in features], dtype=float)

    rng = np.random.default_rng(args.random_state)
    seed_rows = list(rng.choice(len(catalog), size=args.seeds, replace=False))

    baseline_coh: list[float] = []
    weighted_coh: list[float] = []
    affinity_coh: list[float] = []
    baseline_cos: list[float] = []
    weighted_cos: list[float] = []
    affinity_cos: list[float] = []
    baseline_cos_std: list[float] = []
    weighted_cos_std: list[float] = []
    affinity_cos_std: list[float] = []
    baseline_artists: list[float] = []
    weighted_artists: list[float] = []
    affinity_artists: list[float] = []
    availability: list[float] = []

    per_genre: dict[str, dict[str, list[float]]] = {}

    for seed_row in seed_rows:
        group = str(catalog.iloc[seed_row]["recording_group_id"])
        seed_genres = genre_rows[seed_row]
        mask = _same_work_mask(
            str(names.iloc[seed_row]),
            str(artists.iloc[seed_row]),
            names,
            artists,
        )
        mask[seed_row] = True

        result = rec.recommend(group, top_n=TOP_N, include_explanations=False)
        baseline_genres = [_to_genre_set(value) for value in result["genres"]]
        b_coh = _coherence(seed_genres, baseline_genres)
        baseline_coh.append(b_coh)
        baseline_cos.append(float(result["similarity"].mean()))
        baseline_cos_std.append(float(result["similarity"].std()))
        baseline_artists.append(float(result["artists"].nunique() / len(result)))

        w_coh, w_cos, w_std, w_art = _weighted_metrics(
            matrix, unit, seed_row, w, genre_rows, mask, artists
        )
        weighted_coh.append(w_coh)
        weighted_cos.append(w_cos)
        weighted_cos_std.append(w_std)
        weighted_artists.append(w_art)

        a_coh, a_cos, a_std, a_art, avail = _affinity_metrics(
            unit, seed_row, genre_rows, mask, artists
        )
        affinity_coh.append(a_coh)
        affinity_cos.append(a_cos)
        affinity_cos_std.append(a_std)
        affinity_artists.append(a_art)
        availability.append(avail)

        for genre in seed_genres:
            bucket = per_genre.setdefault(
                genre, {"n": 0, "baseline": [], "weighted": [], "affinity": [], "availability": []}
            )
            bucket["n"] += 1
            bucket["baseline"].append(b_coh)
            bucket["weighted"].append(w_coh)
            bucket["affinity"].append(a_coh)
            bucket["availability"].append(avail)

    per_genre_summary = {
        genre: {
            "n_seeds": int(bucket["n"]),
            "availability_in_top100": float(np.mean(bucket["availability"])),
            "baseline_coherence_mean": float(np.mean(bucket["baseline"])),
            "weighted_coherence_mean": float(np.mean(bucket["weighted"])),
            "affinity_coherence_mean": float(np.mean(bucket["affinity"])),
            "affinity_gain": float(np.mean(bucket["affinity"]) - np.mean(bucket["baseline"])),
        }
        for genre, bucket in per_genre.items()
    }

    unrecoverable = sorted(
        [
            genre
            for genre, row in per_genre_summary.items()
            if row["availability_in_top100"] == 0.0 and row["n_seeds"] >= 5
        ]
    )

    report = {
        "experiment": "recommender_genre_coherence",
        "sample_size": args.seeds,
        "random_state": args.random_state,
        "features": features,
        "neighbor_floor": NEIGHBOR_FLOOR,
        "top_n": TOP_N,
        "experiment_weights": EXPERIMENT_WEIGHTS,
        "metrics": {
            "baseline": _variant_report(
                baseline_coh, baseline_cos, baseline_cos_std, baseline_artists
            ),
            "weighted": _variant_report(
                weighted_coh, weighted_cos, weighted_cos_std, weighted_artists
            ),
            "genre_affinity": _variant_report(
                affinity_coh, affinity_cos, affinity_cos_std, affinity_artists, availability
            ),
        },
        "per_genre": per_genre_summary,
        "unrecoverable_genres": unrecoverable,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    metrics = report["metrics"]
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    rows = sorted(
        per_genre_summary.items(), key=lambda item: item[1]["affinity_gain"], reverse=True
    )
    print("\nTop-12 géneros por ganancia de coherencia (afinidad vs baseline):")
    print(f"{'género':<28}{'n':>5}{'avail':>8}{'base':>8}{'afin':>8}{'gan':>8}")
    for genre, row in rows[:12]:
        print(
            f"{genre:<28}{row['n_seeds']:>5}{row['availability_in_top100']:>8.2f}"
            f"{row['baseline_coherence_mean']:>8.2f}{row['affinity_coherence_mean']:>8.2f}"
            f"{row['affinity_gain']:>8.2f}"
        )
    print("\nBottom-12 géneros por ganancia de coherencia:")
    print(f"{'género':<28}{'n':>5}{'avail':>8}{'base':>8}{'afin':>8}{'gan':>8}")
    for genre, row in rows[-12:]:
        print(
            f"{genre:<28}{row['n_seeds']:>5}{row['availability_in_top100']:>8.2f}"
            f"{row['baseline_coherence_mean']:>8.2f}{row['affinity_coherence_mean']:>8.2f}"
            f"{row['affinity_gain']:>8.2f}"
        )
    if unrecoverable:
        print("\nGéneros sin cobertura en el vecindario acústico (n>=5, avail=0):")
        print(", ".join(unrecoverable))
    print(f"\nReporte guardado en {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
