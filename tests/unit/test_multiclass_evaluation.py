import numpy as np
import pytest

from spotify_intelligence.classification.multiclass_evaluation import (
    dominant_genre_exploratory,
    evaluate_multiclass,
    most_confused_pairs,
    normalized_confusion_matrix,
    top_k_accuracy,
)

NAMES = ["rock", "pop", "jazz", "electronic", "classical", "hip-hop"]


def test_top_k_accuracy_perfect():
    y_true = np.array([0, 1, 2])
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
    assert top_k_accuracy(y_true, scores, k=3) == pytest.approx(1.0)


def test_top_k_accuracy_requires_positive_k():
    with pytest.raises(ValueError, match="positive"):
        top_k_accuracy(np.array([0]), np.zeros((1, 3)), k=0)


def test_evaluate_multiclass_perfect():
    y_true = np.array([0, 1, 2])
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
    metrics = evaluate_multiclass(y_true, scores, NAMES[:3])
    for key in (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "confusion_matrix_normalized",
        "most_confused_pairs",
        "n_classes_seen",
    ):
        assert key in metrics
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["top3_accuracy"] == pytest.approx(1.0)


def test_evaluate_multiclass_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="aligned"):
        evaluate_multiclass(np.array([0, 1]), np.zeros((3, 3)), NAMES[:3])


def test_evaluate_multiclass_rejects_column_mismatch():
    with pytest.raises(ValueError, match="class names"):
        evaluate_multiclass(np.array([0, 1]), np.zeros((2, 4)), NAMES[:3])


def test_normalized_confusion_rows_sum_to_one():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 2])
    labels = np.array([0, 1, 2])
    cm = normalized_confusion_matrix(y_true, y_pred, labels)
    np.testing.assert_allclose(cm.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(cm[0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(cm[2], np.array([0.0, 0.0, 1.0]))


def test_most_confused_pairs_sorted():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    pairs = most_confused_pairs(y_true, y_pred, NAMES)
    assert pairs[0]["true_genre"] == "rock"
    assert pairs[0]["predicted_genre"] == "pop"
    assert pairs[0]["count"] == 1
    assert pairs[1]["count"] == 1


def test_dominant_genre_exploratory_metrics():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    scores = np.array([[0.2, 0.1, 0.9], [0.1, 0.9, 0.2]])
    result = dominant_genre_exploratory(y_true, scores, NAMES[:3])
    assert result["hit_at_1"] == pytest.approx(1.0)
    assert result["hit_at_3"] == pytest.approx(1.0)
    assert result["recall_at_5"] == pytest.approx(1.0)
    assert result["rows"] == 2


def test_dominant_genre_exploratory_hit1_misses():
    y_true = np.array([[1, 0, 0]])
    scores = np.array([[0.1, 0.9, 0.0]])
    result = dominant_genre_exploratory(y_true, scores, NAMES[:3])
    assert result["hit_at_1"] == pytest.approx(0.0)
    assert result["hit_at_3"] == pytest.approx(1.0)
    assert result["recall_at_5"] == pytest.approx(1 / 1)
