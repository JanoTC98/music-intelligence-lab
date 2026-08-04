"""Multilabel evaluation metrics and experiment manifests (AGENTS.md sección 16.9).

Scores are uncalibrated; every metric name and report explicitly avoids the
word "probability" unless calibration is proven (sección 16.10).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)

from spotify_intelligence.classification.thresholds import samples_f1


def evaluate_multilabel(
    y_true: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute the sección 16.9 metric set for one evaluation run.

    ``scores`` are the raw uncalibrated model outputs; ``labels`` are the
    thresholded binary predictions. Latency and model size are measured by the
    training/evaluation scripts, not here.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, labels, average="micro", zero_division=0)),
        "samples_f1": float(samples_f1(y_true, labels)),
        "hamming_loss": float(hamming_loss(y_true, labels)),
        "precision_at_3": _precision_at_k(y_true, scores, k=3),
        "recall_at_5": _recall_at_k(y_true, scores, k=5),
        "hit_at_3": _hit_at_k(y_true, scores, k=3),
        "hit_at_5": _hit_at_k(y_true, scores, k=5),
        "coverage_error": _safe_coverage_error(y_true, scores),
        "lrap": _safe_lrap(y_true, scores),
        "per_label_average_precision": _per_label_average_precision(y_true, scores),
    }
    metrics["label_average_precision_mean"] = float(np.mean(metrics["per_label_average_precision"]))
    metrics["macro_precision"] = float(
        precision_score(y_true, labels, average="macro", zero_division=0)
    )
    metrics["macro_recall"] = float(recall_score(y_true, labels, average="macro", zero_division=0))
    return metrics


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    hits = 0.0
    for true_row, score_row in zip(y_true, scores, strict=False):
        top = np.argsort(-score_row)[:k]
        hits += int(true_row[top].sum() > 0)
    return float(hits / len(y_true)) if len(y_true) else 0.0


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    totals = 0.0
    hits = 0.0
    for true_row, score_row in zip(y_true, scores, strict=False):
        top = np.argsort(-score_row)[:k]
        totals += int(true_row.sum())
        hits += int(true_row[top].sum())
    return float(hits / totals) if totals else 0.0


def _hit_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    hit_rows = 0
    for true_row, score_row in zip(y_true, scores, strict=False):
        top = set(np.argsort(-score_row)[:k].tolist())
        if set(np.flatnonzero(true_row).tolist()) & top:
            hit_rows += 1
    return float(hit_rows / len(y_true)) if len(y_true) else 0.0


def _safe_coverage_error(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y_true) == 0 or (y_true.sum(axis=1) == 0).any():
        return None
    return float(coverage_error(y_true, scores))


def _safe_lrap(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y_true) == 0 or (y_true.sum(axis=1) == 0).any():
        return None
    return float(label_ranking_average_precision_score(y_true, scores))


def _per_label_average_precision(y_true: np.ndarray, scores: np.ndarray) -> list[float]:
    values: list[float] = []
    for label in range(y_true.shape[1]):
        column_true = y_true[:, label]
        column_score = scores[:, label]
        if column_true.sum() == 0 or column_true.sum() == len(column_true):
            values.append(float("nan"))
            continue
        values.append(float(average_precision_score(column_true, column_score)))
    return values


def save_metrics_report(
    metrics: dict[str, Any],
    output_dir: str | Path,
    name: str = "multilabel_metrics.json",
) -> Path:
    """Write a metrics report as JSON with NaN serialized as null."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, allow_nan=False)
    return path


def build_experiment_manifest(
    *,
    task: str,
    model_name: str,
    dataset_sha256: str,
    split_sha256: str,
    config_sha256: str,
    git_commit: str | None,
    random_state: int,
    test_used: bool,
    training_seconds: float,
    artifact_path: str,
    started_at_utc: str,
    experiment_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a sección 24.4 experiment manifest."""
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "task": task,
        "model": model_name,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "random_state": random_state,
        "test_used": test_used,
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "training_seconds": round(training_seconds, 4),
        "artifact_path": artifact_path,
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    """Write an experiment manifest as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def measure_prediction_latency(
    predict_fn: Callable[[np.ndarray], Any],
    X: np.ndarray,
    *,
    repeat: int = 3,
) -> float:
    """Return the mean prediction latency in milliseconds over ``repeat`` runs."""
    start = time.perf_counter()
    for _ in range(repeat):
        predict_fn(X)
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(repeat, 1)
    return round(float(elapsed_ms), 4)


def model_size_mb(path: str | Path) -> float:
    """Return the serialized model size in MB (recursively for directories)."""
    target = Path(path)
    if target.is_dir():
        total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    else:
        total = target.stat().st_size
    return round(total / (1024 * 1024), 4)
