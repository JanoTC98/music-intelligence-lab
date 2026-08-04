"""Prediction helpers for the multilabel classifier (AGENTS.md sección 16.8).

The application always shows Top-5 genres. If no label surpasses the tuned
global threshold, the top-1 genre is returned with a ``below_threshold`` flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spotify_intelligence.classification.thresholds import apply_threshold
from spotify_intelligence.features.encoders import GenreLabelEncoder

DEFAULT_TOP_K = 5


def top_k_genres(
    scores: np.ndarray,
    encoder: GenreLabelEncoder,
    *,
    k: int = DEFAULT_TOP_K,
) -> list[list[str]]:
    """Return the top-k genre names for each row of a score matrix."""
    result: list[list[str]] = []
    for row in np.asarray(scores):
        order = np.argsort(-row)[:k]
        result.append([encoder.classes_[idx] for idx in order])
    return result


def predict_with_threshold(
    scores: np.ndarray,
    encoder: GenreLabelEncoder,
    threshold: float,
    *,
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Apply the global threshold and return Top-k plus a below-threshold flag.

    Returns a dict with ``labels`` (binary matrix), ``top_k_genres``,
    ``below_threshold`` (one flag per row) and ``scores`` unchanged.
    """
    labels = apply_threshold(scores, threshold)
    below_threshold = (labels.sum(axis=1) == 0).astype(bool)
    return {
        "labels": labels,
        "top_k_genres": top_k_genres(scores, encoder, k=k),
        "below_threshold": below_threshold,
        "scores": np.asarray(scores),
    }
