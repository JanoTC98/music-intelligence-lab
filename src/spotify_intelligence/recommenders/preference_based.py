"""Content-based preference recommender (AGENTS.md sección 15).

Ranking uses a weighted Euclidean distance over scaled features. Presets come
from ``configs/presets.yaml`` and remain editable without touching Python.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from spotify_intelligence.features.presets import (
    BASIC_FEATURES,
    VALUE_RANGES,
    get_preset,
)
from spotify_intelligence.recommenders.errors import (
    ArtifactNotFoundError,
    InvalidPreferenceProfileError,
)
from spotify_intelligence.recommenders.track_based import RecommendationFilters

ARTIFACT_FILES = (
    "scaler.joblib",
    "catalog_matrix.npy",
    "catalog_index.parquet",
    "ood_reference.json",
)

WEIGHT_MIN = 0
WEIGHT_MAX = 3


def weighted_euclidean_distance(
    point: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Euclidean distance over scaled variables (sección 15.6).

    ``d = sqrt(sum(w_i * (x_i - q_i)^2) / sum(w_i))``. All arrays must be
    one-dimensional and share the same length.
    """
    if point.shape != query.shape or query.shape != weights.shape:
        raise ValueError("point, query and weights must share the same shape")
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise InvalidPreferenceProfileError("Cannot compute a distance with all weights zero")
    squared = weights * ((point - query) ** 2)
    return float(math.sqrt(np.sum(squared) / total_weight))


@dataclass(frozen=True)
class PreferenceProfile:
    """A user-editable acoustic profile (preset or manual mode)."""

    values: dict[str, float]
    weights: dict[str, int]
    label: str | None = None

    def __post_init__(self) -> None:
        _validate_profile(self.values, self.weights)

    @classmethod
    def from_preset(
        cls,
        key: str,
        presets_path: str | Path = "configs/presets.yaml",
    ) -> PreferenceProfile:
        preset = get_preset(key, presets_path)
        return cls(values=preset["values"], weights=preset["weights"], label=preset["label"])

    @classmethod
    def from_manual(
        cls,
        values: dict[str, float],
        weights: dict[str, int],
        label: str | None = None,
    ) -> PreferenceProfile:
        return cls(values=dict(values), weights=dict(weights), label=label)

    @property
    def query_vector(self) -> np.ndarray:
        """Raw (unscaled) query vector aligned to ``BASIC_FEATURES``."""
        return np.asarray([float(self.values[f]) for f in BASIC_FEATURES], dtype=float)

    @property
    def weight_vector(self) -> np.ndarray:
        """Weight vector aligned to ``BASIC_FEATURES``; missing weights become 0."""
        return np.asarray([float(self.weights.get(f, 0)) for f in BASIC_FEATURES], dtype=float)


def _validate_profile(values: dict[str, float], weights: dict[str, int]) -> None:
    unknown = set(weights) - set(BASIC_FEATURES)
    if unknown:
        raise InvalidPreferenceProfileError(f"Unknown weighted features: {sorted(unknown)}")

    for feature, weight in weights.items():
        if not WEIGHT_MIN <= weight <= WEIGHT_MAX:
            raise InvalidPreferenceProfileError(
                f"Weight for {feature!r} out of range [{WEIGHT_MIN}, {WEIGHT_MAX}]"
            )
        if weight > 0 and feature not in values:
            raise InvalidPreferenceProfileError(f"Missing value for weighted feature {feature!r}")

    if all(w == 0 for w in weights.values()):
        raise InvalidPreferenceProfileError("All weights are zero; the profile is not usable")

    for feature, value in values.items():
        if feature in VALUE_RANGES:
            low, high = VALUE_RANGES[feature]
            if not low <= float(value) <= high:
                raise InvalidPreferenceProfileError(
                    f"Value for {feature!r} out of range [{low}, {high}]"
                )


class PreferenceRecommender:
    """Preference recommender backed by versioned artifacts.

    The catalog matrix is already scaled by the stored scaler. OOD warnings use
    the distance to the catalog centroid (percentiles p95/p99) computed at build.
    """

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self._check_artifacts()
        self.scaler = joblib.load(self.artifact_dir / "scaler.joblib")
        self.catalog_matrix = np.load(self.artifact_dir / "catalog_matrix.npy")
        self.catalog_index = pd.read_parquet(self.artifact_dir / "catalog_index.parquet")
        self.ood_reference = self._load_ood_reference()
        self.manifest = self._load_manifest()
        self._row_to_group = self.catalog_index["recording_group_id"].tolist()

    def _check_artifacts(self) -> None:
        missing = [name for name in ARTIFACT_FILES if not (self.artifact_dir / name).exists()]
        if missing:
            raise ArtifactNotFoundError(
                f"Missing preference recommender artifacts in {self.artifact_dir}: {missing}"
            )

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise ArtifactNotFoundError(f"Missing preference recommender manifest: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_ood_reference(self) -> dict[str, Any]:
        path = self.artifact_dir / "ood_reference.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def recommend(
        self,
        profile: PreferenceProfile,
        *,
        top_n: int = 10,
        filters: RecommendationFilters | None = None,
        diversity_enabled: bool = False,
        lambda_: float = 0.85,
    ) -> pd.DataFrame:
        """Rank the catalog by weighted distance to ``profile`` and return Top-N."""
        filters = filters or RecommendationFilters()
        scaled_query = self.scaler.transform(profile.query_vector.reshape(1, -1))[0]
        weights = profile.weight_vector

        distances = np.asarray(
            [weighted_euclidean_distance(row, scaled_query, weights) for row in self.catalog_matrix]
        )

        order = np.argsort(distances, kind="stable")
        distances_sorted = distances[order]
        rows = [int(i) for i in order]
        rows, distances_sorted = self._apply_filters(rows, distances_sorted, filters)
        if not rows:
            return self._empty_results()

        similarities = [1.0 / (1.0 + d) for d in distances_sorted]
        results = self.catalog_index.iloc[rows].copy()
        results["distance"] = distances_sorted
        results["similarity"] = similarities
        results = results.drop_duplicates(subset=["recording_group_id"], keep="first")
        results = results.sort_values(
            ["distance", "recording_group_id"], ascending=[True, True]
        ).head(top_n)
        results = results.reset_index(drop=True)

        if diversity_enabled and len(results) > 1:
            results = self._apply_diversity(results, profile, lambda_=lambda_)

        return results.reset_index(drop=True)

    def _apply_filters(
        self,
        rows: list[int],
        distances: list[float],
        filters: RecommendationFilters,
    ) -> tuple[list[int], list[float]]:
        if not rows:
            return [], []
        df = self.catalog_index.iloc[rows].copy()
        df["_dist"] = distances

        if filters.explicit == "explicit":
            df = df[df["explicit"]]
        elif filters.explicit == "non_explicit":
            df = df[~df["explicit"]]

        if filters.genres:
            allowed = set(filters.genres)
            df = df[df["genres"].map(lambda genres: bool(set(genres) & allowed))]

        if filters.duration_enabled:
            min_ms = filters.duration_min_seconds * 1000
            max_ms = filters.duration_max_seconds * 1000
            df = df[(df["duration_ms"] >= min_ms) & (df["duration_ms"] <= max_ms)]

        if filters.popularity_min is not None:
            df = df[df["popularity_median"] >= filters.popularity_min]

        if len(df) == 0:
            return [], []
        return df.index.tolist(), df["_dist"].tolist()

    def _apply_diversity(
        self,
        results: pd.DataFrame,
        profile: PreferenceProfile,
        *,
        lambda_: float,
    ) -> pd.DataFrame:
        from spotify_intelligence.recommenders.diversity import mmr_rerank

        n = len(results)
        weights = profile.weight_vector
        raw_rows = [self._row_to_group.index(g) for g in results["recording_group_id"]]
        vectors = self.catalog_matrix[raw_rows]

        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarities[i, j] = 1.0 / (
                        1.0 + weighted_euclidean_distance(vectors[i], vectors[j], weights)
                    )

        relevance = results["similarity"].to_numpy(dtype=float)
        order = mmr_rerank(relevance, similarities, lambda_=lambda_)
        return results.iloc[order].reset_index(drop=True)

    def out_of_distribution_status(
        self,
        profile: PreferenceProfile,
    ) -> dict[str, Any]:
        """Compare the profile distance to the catalog centroid with p95/p99."""
        scaled_query = self.scaler.transform(profile.query_vector.reshape(1, -1))[0]
        centroid = np.asarray(self.ood_reference["centroid"], dtype=float)
        distance = float(np.linalg.norm(scaled_query - centroid))
        p95 = float(self.ood_reference["percentiles"]["95"])
        p99 = float(self.ood_reference["percentiles"]["99"])

        if distance > p99:
            status = "weak_match"
        elif distance > p95:
            status = "warning"
        else:
            status = "ok"
        return {
            "status": status,
            "distance_to_centroid": distance,
            "p95": p95,
            "p99": p99,
        }

    def _empty_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "recording_group_id",
                "representative_track_id",
                "track_name",
                "artists",
                "album_name",
                "genres",
                "duration_ms",
                "popularity_median",
                "explicit",
                "distance",
                "similarity",
            ]
        )
