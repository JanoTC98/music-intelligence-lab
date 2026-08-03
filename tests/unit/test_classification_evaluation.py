import numpy as np
import pytest

from spotify_intelligence.classification.evaluation import (
    _per_label_average_precision,
    build_experiment_manifest,
    evaluate_multilabel,
    measure_prediction_latency,
    model_size_mb,
    save_manifest,
    save_metrics_report,
)


def test_evaluate_multilabel_returns_expected_keys():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3], [0.8, 0.6, 0.1]])
    labels = (scores >= 0.5).astype(np.int8)
    metrics = evaluate_multilabel(y_true, scores, labels)

    for key in (
        "macro_f1",
        "micro_f1",
        "samples_f1",
        "hamming_loss",
        "precision_at_3",
        "recall_at_5",
        "hit_at_3",
        "hit_at_5",
        "coverage_error",
        "lrap",
        "per_label_average_precision",
        "label_average_precision_mean",
        "macro_precision",
        "macro_recall",
    ):
        assert key in metrics


def test_perfect_predictions_hit_at_k_one():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
    labels = (scores >= 0.5).astype(np.int8)
    metrics = evaluate_multilabel(y_true, scores, labels)
    assert metrics["hit_at_3"] == pytest.approx(1.0)
    assert metrics["samples_f1"] == pytest.approx(1.0)
    assert metrics["hamming_loss"] == pytest.approx(0.0)


def test_coverage_and_lrap_none_when_row_empty():
    y_true = np.array([[1, 0], [0, 0]])
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = (scores >= 0.5).astype(np.int8)
    metrics = evaluate_multilabel(y_true, scores, labels)
    assert metrics["coverage_error"] is None
    assert metrics["lrap"] is None


def test_per_label_average_precision_skips_constant_columns():
    y_true = np.array([[1, 0, 0], [1, 0, 1]])
    scores = np.array([[0.9, 0.5, 0.1], [0.8, 0.5, 0.9]])
    values = _per_label_average_precision(y_true, scores)
    assert np.isnan(values[0])  # columna constante -> NaN


def test_save_metrics_report_writes_json(tmp_path):
    report = {"samples_f1": 0.5}
    path = save_metrics_report(report, tmp_path, name="custom.json")
    assert path.exists()
    assert '"samples_f1": 0.5' in path.read_text(encoding="utf-8")


def test_save_manifest_roundtrip(tmp_path):
    manifest = build_experiment_manifest(
        task="multilabel",
        model_name="test",
        dataset_sha256="a",
        split_sha256="b",
        config_sha256="c",
        git_commit=None,
        random_state=42,
        test_used=False,
        training_seconds=1.0,
        artifact_path="models/x",
        started_at_utc="2026-01-01T00:00:00+00:00",
        experiment_id="exp",
    )
    path = save_manifest(manifest, tmp_path / "artifacts")
    assert path.exists()
    assert manifest["test_used"] is False


def test_model_size_mb_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\x00" * (2 * 1024 * 1024))
    assert model_size_mb(tmp_path) == pytest.approx(2.0)


def test_measure_prediction_latency_positive():
    latency = measure_prediction_latency(lambda x: x, np.zeros((10, 3)))
    assert latency >= 0
