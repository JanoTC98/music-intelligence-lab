"""Multiclass (dominant genre) classification models (AGENTS.md sección 17).

Predicts a single dominant genre from the primary audio features. The target
``y`` is the genre index into the canonical 114-genre encoder; scikit-learn
models learn a dense ``classes_`` subset (only the classes present in train).

Scores are uncalibrated; they are never called probabilities (sección 16.10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from spotify_intelligence.config import load_yaml_config

MODEL_IDS = {
    "C0": "frequent_class_baseline",
    "C1": "logistic",
    "C2": "extra_trees",
    "C3": "random_forest",
    "C4": "xgboost",
}


class FrequentClassBaseline:
    """C0: always predict the most frequent class observed in training.

    ``classes_`` mimics the scikit-learn attribute: the sorted unique class
    values seen at fit time (genre indices). ``predict_proba`` returns a
    one-hot row over ``classes_`` so the score matrix stays comparable.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.majority_index_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> FrequentClassBaseline:
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = classes.astype(int)
        self.majority_index_ = int(np.argmax(counts))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("FrequentClassBaseline must be fitted before predict")
        return np.full(len(X), self.majority_index_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("FrequentClassBaseline must be fitted before predict_proba")
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        proba[:, self.majority_index_] = 1.0
        return proba


def _drop_enabled(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k != "enabled"}


def _build_logistic(params: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(**_drop_enabled(params))


def _build_extra_trees(params: dict[str, Any]) -> ExtraTreesClassifier:
    extra = {k: v for k, v in params.items() if k not in ("enabled", "n_jobs")}
    return ExtraTreesClassifier(n_jobs=params.get("n_jobs", -1), **extra)


def _build_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    extra = {k: v for k, v in params.items() if k not in ("enabled", "n_jobs")}
    return RandomForestClassifier(n_jobs=params.get("n_jobs", -1), **extra)


def build_model(model_id: str, params: dict[str, Any]) -> Any:
    """Instantiate a multiclass model by its AGENTS.md id (C0..C4)."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model id: {model_id}")
    key = MODEL_IDS[model_id]
    config = params["multiclass"][key]
    if model_id == "C0":
        return FrequentClassBaseline()
    if model_id == "C1":
        return _build_logistic(config)
    if model_id == "C2":
        return _build_extra_trees(config)
    if model_id == "C3":
        return _build_random_forest(config)
    if model_id == "C4":
        raise ValueError("XGBoost (C4) is optional and disabled in this project")
    raise ValueError(f"Unknown model id: {model_id}")


def predict_proba_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Return the dense ``[n, n_classes_seen]`` score matrix of a model."""
    return np.asarray(model.predict_proba(X), dtype=float)


def expand_to_full_label_space(
    dense_scores: np.ndarray,
    classes_: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Map dense per-class scores into the full 114-label space.

    ``classes_`` holds the genre indices that the model actually learned; the
    remaining labels get a score of zero.
    """
    dense_scores = np.asarray(dense_scores)
    if dense_scores.ndim != 2:
        raise ValueError("dense_scores must be a 2D matrix")
    full = np.zeros((dense_scores.shape[0], n_labels), dtype=float)
    full[:, np.asarray(classes_, dtype=int)] = dense_scores
    return full


def model_classes(model: Any) -> np.ndarray:
    """Return the model ``classes_`` as an int array of genre indices."""
    return np.asarray(model.classes_, dtype=int)


def load_model_parameters(path: str | Path = "configs/model_parameters.yaml") -> dict[str, Any]:
    """Load the classifier hyperparameter configuration."""
    return load_yaml_config(path)
