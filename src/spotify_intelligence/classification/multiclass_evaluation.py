"""Multiclass (dominant genre) evaluation metrics (AGENTS.md sección 17.4, sección 17.5).

All score matrices are in the full 114-genre space (see
``expand_to_full_label_space``). ``y_true`` is a 1D array of genre indices
aligned to the canonical 114-genre encoder. Scores are uncalibrated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def top_k_accuracy(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of rows whose true class appears in the top-k scores."""
    if k <= 0:
        raise ValueError("k must be positive")
    hits = 0
    for true_row, score_row in zip(np.asarray(y_true), np.asarray(scores), strict=False):
        top = set(np.argsort(-score_row)[:k].tolist())
        if int(true_row) in top:
            hits += 1
    return float(hits / len(y_true)) if len(y_true) else 0.0


def normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Confusion matrix normalized by row (true) counts."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)


def most_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Off-diagonal confusion pairs sorted by count, as genre-name records."""
    cm = confusion_matrix(y_true, y_pred)
    pairs: list[dict[str, Any]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true_genre": str(class_names[i]),
                        "predicted_genre": str(class_names[j]),
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_n]


def evaluate_multiclass(
    y_true: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute the sección 17.4 metric set for one evaluation run.

    ``scores`` must have one column per ``class_names`` entry (full label
    space). Latency, size and training time are measured by the scripts.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != y_true.shape[0]:
        raise ValueError("scores must be a [n_rows, n_labels] matrix aligned to y_true")
    if scores.shape[1] != len(class_names):
        raise ValueError("scores columns must equal the number of class names")
    y_pred = scores.argmax(axis=1)
    labels = np.arange(len(class_names), dtype=int)
    cm_norm = normalized_confusion_matrix(y_true, y_pred, labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy(y_true, scores, k=3)),
        "top5_accuracy": float(top_k_accuracy(y_true, scores, k=5)),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "most_confused_pairs": most_confused_pairs(y_true, y_pred, class_names),
        "n_classes_seen": int(np.unique(y_true).size),
    }


def dominant_genre_exploratory(
    y_true_multilabel: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    *,
    k_hit1: int = 1,
    k_hit3: int = 3,
    k_recall5: int = 5,
) -> dict[str, Any]:
    """Exploratory evaluation on multi-genre rows (sección 17.5).

    Applies a multiclass score matrix to rows whose original label set has
    more than one genre and reports Hit@1, Hit@3 and Recall@5 against that
    full original set. The predicted dominant genre never replaces labels.
    """
    y_true = np.asarray(y_true_multilabel)
    scores = np.asarray(scores)
    if y_true.ndim != 2 or scores.ndim != 2:
        raise ValueError("y_true and scores must be 2D matrices")
    if scores.shape[1] != len(class_names):
        raise ValueError("scores columns must equal the number of class names")

    hit1 = hit3 = 0
    recall_hits = recall_total = 0
    for true_row, score_row in zip(y_true, scores, strict=False):
        original = set(np.flatnonzero(true_row).tolist())
        order = np.argsort(-score_row)
        if int(order[0]) in original:
            hit1 += 1
        if set(order[:k_hit3].tolist()) & original:
            hit3 += 1
        recall_hits += len(set(order[:k_recall5].tolist()) & original)
        recall_total += len(original)

    n = len(y_true)
    return {
        "hit_at_1": float(hit1 / n) if n else 0.0,
        "hit_at_3": float(hit3 / n) if n else 0.0,
        "recall_at_5": float(recall_hits / recall_total) if recall_total else 0.0,
        "rows": int(n),
    }
