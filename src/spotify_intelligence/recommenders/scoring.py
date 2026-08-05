from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from spotify_intelligence.recommenders.errors import IncompatibleArtifactError

SCALER_FACTORY: dict[str, type] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
}

RECOMMENDER_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)


def create_scaler(name: str):
    """Instantiate a scaler by its configured name."""
    if name not in SCALER_FACTORY:
        raise IncompatibleArtifactError(f"Unsupported scaler: {name!r}")
    return SCALER_FACTORY[name]()


def fit_scaler(
    recordings: pd.DataFrame,
    features: tuple[str, ...] = RECOMMENDER_FEATURES,
    scaler_name: str = "standard",
):
    """Fit a scaler on the eligible recordings feature matrix."""
    matrix = recordings[list(features)].to_numpy(dtype=float)
    scaler = create_scaler(scaler_name)
    scaler.fit(matrix)
    return scaler


def transform_matrix(
    matrix: np.ndarray,
    scaler,
) -> np.ndarray:
    """Scale a raw feature matrix with an already fitted scaler."""
    return scaler.transform(matrix)  # type: ignore[no-any-return]


def build_eligible_recordings(
    recordings: pd.DataFrame,
    *,
    exclude_incomplete: bool = True,
) -> pd.DataFrame:
    """Return the recordings eligible for recommendation."""
    result = recordings
    if exclude_incomplete:
        result = result[~result["audio_analysis_incomplete"]]
    return result.reset_index(drop=True)


def cosine_similarity_from_distance(distance: float) -> float:
    """Convert a cosine distance to a similarity score.

    ``similarity = 1 - cosine_distance``. This is NOT a probability.
    """
    return float(1.0 - distance)


def prepare_catalog_matrix(
    recordings: pd.DataFrame,
    features: tuple[str, ...] = RECOMMENDER_FEATURES,
    scaler_name: str = "standard",
) -> tuple[np.ndarray, Any]:
    """Fit a scaler on the catalog and return the scaled matrix plus the scaler."""
    matrix = recordings[list(features)].to_numpy(dtype=float)
    scaler = create_scaler(scaler_name)
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler
