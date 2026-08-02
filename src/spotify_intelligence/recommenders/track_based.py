from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from spotify_intelligence.recommenders.errors import ArtifactNotFoundError
from spotify_intelligence.recommenders.scoring import cosine_similarity_from_distance

ARTIFACT_FILES = (
    "scaler.joblib",
    "neighbors.joblib",
    "catalog_matrix.npy",
    "catalog_index.parquet",
)


@dataclass
class RecommendationFilters:
    """User-configurable filters applied after the nearest-neighbor search.

    Defaults match AGENTS.md §14.8: all filters are inactive unless enabled.
    """

    explicit: str = "all"
    genres: list[str] | None = None
    duration_enabled: bool = False
    duration_min_seconds: int = 60
    duration_max_seconds: int = 600
    different_artist: bool = False
    popularity_min: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RecommendationFilters:
        filters_config = config["track_recommender"]["filters"]
        return cls(
            explicit=str(filters_config.get("explicit_default", "all")),
            duration_enabled=bool(filters_config.get("duration_enabled_default", False)),
            duration_min_seconds=int(filters_config.get("duration_suggested_min_seconds", 60)),
            duration_max_seconds=int(filters_config.get("duration_suggested_max_seconds", 600)),
            different_artist=bool(filters_config.get("different_artist_default", False)),
        )


class TrackRecommender:
    """Content-based track recommender backed by a versioned set of artifacts."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self._check_artifacts()
        self.scaler = joblib.load(self.artifact_dir / "scaler.joblib")
        self.neighbors = joblib.load(self.artifact_dir / "neighbors.joblib")
        self.catalog_matrix = np.load(self.artifact_dir / "catalog_matrix.npy")
        self.catalog_index = pd.read_parquet(self.artifact_dir / "catalog_index.parquet")
        self.manifest = self._load_manifest()
        self._row_to_group = self.catalog_index["recording_group_id"].tolist()
        self._group_to_row = {group: row for row, group in enumerate(self._row_to_group)}

    def _check_artifacts(self) -> None:
        missing = [name for name in ARTIFACT_FILES if not (self.artifact_dir / name).exists()]
        if missing:
            raise ArtifactNotFoundError(
                f"Missing recommender artifacts in {self.artifact_dir}: {missing}"
            )

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise ArtifactNotFoundError(f"Missing recommender manifest: {manifest_path}")
        import json

        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.manifest.get("features", []))

    def _seed_row_position(self, recording_group_id: str) -> int:
        try:
            return self._group_to_row[recording_group_id]
        except KeyError:
            raise ArtifactNotFoundError(
                f"Recording group {recording_group_id} is not in the recommender catalog"
            ) from None

    def recommend(
        self,
        recording_group_id: str,
        *,
        top_n: int = 10,
        filters: RecommendationFilters | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with the Top-N recommendations for a seed recording."""
        filters = filters or RecommendationFilters()
        seed_row = self._seed_row_position(recording_group_id)
        query = self.catalog_matrix[seed_row : seed_row + 1]

        candidate_rows, distances = self._retrieve_candidates(query, seed_row, top_n, filters)
        if len(candidate_rows) == 0:
            return self._empty_results()

        similarities = [cosine_similarity_from_distance(d) for d in distances]
        results = self.catalog_index.iloc[candidate_rows].copy()
        results["distance"] = distances
        results["similarity"] = similarities
        results = self._deduplicate_groups(results)
        results = results.sort_values(["similarity", "recording_group_id"], ascending=[False, True])
        return results.head(top_n).reset_index(drop=True)

    def _retrieve_candidates(
        self,
        query: np.ndarray,
        seed_row: int,
        top_n: int,
        filters: RecommendationFilters,
    ) -> tuple[list[int], list[float]]:
        config = self.manifest.get("retrieval", {})
        floor = int(config.get("initial_candidate_floor", 100))
        multiplier = int(config.get("candidate_multiplier", 10))
        expansion_steps = list(config.get("expansion_steps", [500, 2000]))

        k_values = [max(floor, top_n * multiplier), *expansion_steps]
        for k in k_values:
            if k >= self.catalog_matrix.shape[0]:
                k = self.catalog_matrix.shape[0]
                k_values = [k]
                break

        for k in k_values:
            n_neighbors = min(k + 1, self.catalog_matrix.shape[0])
            distances, indices = self.neighbors.kneighbors(query, n_neighbors=n_neighbors)
            rows = [int(i) for i in indices[0] if int(i) != seed_row]
            dists = [
                float(d)
                for i, d in zip(indices[0], distances[0], strict=False)
                if int(i) != seed_row
            ]
            rows, dists = self._apply_filters(rows, dists, seed_row, filters)
            if len(rows) >= top_n or k == self.catalog_matrix.shape[0]:
                return rows, dists
        return [], []

    def _apply_filters(
        self,
        rows: list[int],
        distances: list[float],
        seed_row: int,
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

        if filters.different_artist:
            seed_artists = self.catalog_index.iloc[seed_row]["artists"]
            df = df[df["artists"] != seed_artists]

        if len(df) == 0:
            return [], []
        return df.index.tolist(), df["_dist"].tolist()

    def _deduplicate_groups(self, results: pd.DataFrame) -> pd.DataFrame:
        return results.drop_duplicates(subset=["recording_group_id"], keep="first")

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
