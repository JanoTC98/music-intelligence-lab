import numpy as np
import pytest

from spotify_intelligence.classification.thresholds import (
    apply_threshold,
    samples_f1,
    tune_global_threshold,
)


def test_samples_f1_perfect_predictions():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = y_true.copy()
    assert samples_f1(y_true, y_pred) == pytest.approx(1.0)


def test_samples_f1_no_true_labels_scores_zero():
    y_true = np.zeros((2, 3), dtype=np.int8)
    y_pred = np.zeros((2, 3), dtype=np.int8)
    assert samples_f1(y_true, y_pred) == 0.0


def test_tune_global_threshold_recovers_separated_scores():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.95, 0.05], [0.05, 0.95]])
    result = tune_global_threshold(scores, y_true)
    assert result.best_score == pytest.approx(1.0)


def test_tune_global_threshold_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must share shape"):
        tune_global_threshold(np.zeros((2, 3)), np.zeros((2, 4)))


def test_threshold_curve_within_grid_bounds():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.9]])
    result = tune_global_threshold(scores, y_true)
    assert all(0.10 <= step["threshold"] <= 0.90 for step in result.curve)
    assert len(result.curve) == 17


def test_apply_threshold_binary():
    scores = np.array([[0.9, 0.05, 0.5]])
    labels = apply_threshold(scores, 0.5)
    np.testing.assert_array_equal(labels, np.array([[1, 0, 1]]))


def test_tune_global_threshold_prefers_higher_boundary():
    y_true = np.array([[1, 0, 0]])
    scores = np.array([[0.85, 0.6, 0.4]])
    result = tune_global_threshold(scores, y_true)
    assert result.best_threshold > 0.6
