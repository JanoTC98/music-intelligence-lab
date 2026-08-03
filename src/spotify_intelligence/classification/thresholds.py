"""Global threshold tuning for multilabel predictions (AGENTS.md §16.8).

A single global threshold converts per-label scores into binary predictions.
The threshold is tuned on the validation split only, optimizing samples F1.
Test data is never used for tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

THRESHOLD_GRID = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))


@dataclass
class ThresholdResult:
    """Best threshold and the score curve evaluated on validation."""

    best_threshold: float
    best_score: float
    optimize_metric: str
    curve: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def samples_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute samples F1 (harmonic mean per row, averaged across rows)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = (y_true & y_pred).sum(axis=1).astype(float)
    fp = (y_pred & ~y_true).sum(axis=1).astype(float)
    fn = (y_true & ~y_pred).sum(axis=1).astype(float)
    denom = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(scores.mean()) if len(scores) else 0.0


def tune_global_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    grid: tuple[float, ...] = THRESHOLD_GRID,
    metric: Callable[[np.ndarray, np.ndarray], float] = samples_f1,
) -> ThresholdResult:
    """Return the global threshold that maximizes ``metric`` on ``y_true``.

    ``scores`` is a ``[n, n_labels]`` score matrix (uncalibrated). The grid
    starts at 0.10 because labels below that are never selected (§16.8).
    """
    if scores.shape != y_true.shape:
        raise ValueError("scores and y_true must share shape")
    curve: list[dict[str, float]] = []
    best_threshold, best_score = grid[0], -np.inf
    for threshold in grid:
        preds = (scores >= threshold).astype(np.int8)
        score = metric(y_true, preds)
        curve.append({"threshold": threshold, "score": float(score)})
        if score > best_score:
            best_threshold, best_score = threshold, score
    return ThresholdResult(
        best_threshold=float(best_threshold),
        best_score=float(best_score),
        optimize_metric="samples_f1",
        curve=curve,
    )


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize a score matrix using the tuned global threshold."""
    return (np.asarray(scores) >= threshold).astype(np.int8)
