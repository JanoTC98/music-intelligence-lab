"""Multilabel classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

from spotify_intelligence.config import load_yaml_config

MODEL_IDS = {
    "M0": "frequency_baseline",
    "M1": "ovr_logistic",
    "M2": "classifier_chain",
    "M3": "extra_trees",
    "M4": "random_forest",
    "M5": "xgboost_ovr",
}


class FrequencyBaseline:
    """M0: predict the most frequent labels without using features.

    ``predict_proba`` returns the observed per-label prevalence from training
    for every row. This estimator is deliberately feature-independent.
    """

    def __init__(self) -> None:
        self.label_frequencies_: np.ndarray | None = None
        self.n_labels_ = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> FrequencyBaseline:
        self.n_labels_ = int(Y.shape[1])
        self.label_frequencies_ = Y.mean(axis=0).astype(float)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.label_frequencies_ is None:
            raise RuntimeError("FrequencyBaseline must be fitted before predict_proba")
        return np.tile(self.label_frequencies_, (X.shape[0], 1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int8)


def _build_ovr_logistic(params: dict[str, Any]) -> OneVsRestClassifier:
    estimator_params = {k: v for k, v in params.items() if k != "wrapper_n_jobs"}
    estimator = LogisticRegression(**estimator_params)
    n_jobs = params.get("wrapper_n_jobs", -1)
    return OneVsRestClassifier(estimator, n_jobs=n_jobs)


def _build_classifier_chain(params: dict[str, Any], random_state: int = 42) -> ClassifierChain:
    base = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    return ClassifierChain(base, order="random", random_state=random_state)


class ClassifierChainEnsemble:
    """M2: an ensemble of ClassifierChain models with averaged scores.

    Three chains are trained with ``order="random"`` and seeds ``42, 43, 44``
    Scores are averaged for comparison; the prediction is derived
    from the averaged score matrix. Order dependence is documented as an
    experimental limitation.
    """

    def __init__(self, seeds: list[int] | None = None) -> None:
        self.seeds = seeds or [42, 43, 44]
        self.chains_: list[ClassifierChain] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ClassifierChainEnsemble:
        for seed in self.seeds:
            chain = _build_classifier_chain({}, random_state=seed)
            chain.fit(X, Y)
            self.chains_.append(chain)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.chains_:
            raise RuntimeError("ClassifierChainEnsemble must be fitted before predict_proba")
        scores = [np.asarray(chain.predict_proba(X), dtype=float) for chain in self.chains_]
        return np.mean(scores, axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int8)


def _build_extra_trees(params: dict[str, Any]) -> ExtraTreesClassifier:
    extra = {k: v for k, v in params.items() if k not in ("enabled", "n_jobs")}
    return ExtraTreesClassifier(n_jobs=params.get("n_jobs", -1), **extra)


def _build_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    extra = {k: v for k, v in params.items() if k not in ("enabled", "n_jobs")}
    return RandomForestClassifier(n_jobs=params.get("n_jobs", -1), **extra)


def build_model(model_id: str, params: dict[str, Any]) -> Any:
    """Instantiate a multilabel model by its id (M0..M5)."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model id: {model_id}")
    key = MODEL_IDS[model_id]
    config = params["multilabel"][key]
    if model_id == "M0":
        return FrequencyBaseline()
    if model_id == "M1":
        return _build_ovr_logistic(
            config["estimator"] | {"wrapper_n_jobs": config["wrapper_n_jobs"]}
        )
    if model_id == "M2":
        return ClassifierChainEnsemble(seeds=config.get("seeds", [42, 43, 44]))
    if model_id == "M3":
        return _build_extra_trees(config)
    if model_id == "M4":
        return _build_random_forest(config)
    if model_id == "M5":
        raise ValueError("XGBoost (M5) is optional and disabled in this project")
    raise ValueError(f"Unknown model id: {model_id}")


def predict_proba_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Return a ``[n, n_labels]`` score matrix for any supported model.

    OneVsRest and ClassifierChain expose ``predict_proba`` directly. Tree
    multi-output classifiers return a list of per-output arrays; those are
    stacked into the positive-class probability per label.
    """
    proba = model.predict_proba(X)
    if isinstance(proba, np.ndarray):
        return np.asarray(proba, dtype=float)
    if isinstance(proba, list) and proba and isinstance(proba[0], np.ndarray):
        n_labels = len(proba)
        matrix = np.column_stack([p[:, 1] if p.shape[1] == 2 else p[:, 0] for p in proba])
        assert matrix.shape[1] == n_labels, "score matrix columns must equal n_labels"
        return matrix
    raise TypeError(f"Unsupported predict_proba return type: {type(proba)}")


def load_model_parameters(
    path: str | Path = "configs/model_parameters.yaml",
) -> dict[str, Any]:
    """Load the classifier hyperparameter configuration."""
    return load_yaml_config(path)
