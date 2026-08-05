import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.datasets import build_multilabel_dataset  # noqa: E402
from spotify_intelligence.classification.multilabel import (  # noqa: E402
    ClassifierChainEnsemble,
    FrequencyBaseline,
    build_model,
    load_model_parameters,
    predict_proba_scores,
)
from spotify_intelligence.classification.training import (  # noqa: E402
    TIME_SIGNATURE_CATEGORIES,
    feature_matrix,
)
from spotify_intelligence.features.audio_features import INCOMPLETE_AUDIO_COLUMNS  # noqa: E402


@pytest.fixture()
def dataset():
    fixture = make_classification_fixture()
    return build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )


def test_frequency_baseline_uses_prevalence(dataset):
    X = dataset.to_matrix()
    model = FrequencyBaseline().fit(X, dataset.Y)
    proba = model.predict_proba(X)
    assert proba.shape == dataset.Y.shape
    expected = dataset.Y.mean(axis=0)
    np.testing.assert_allclose(proba[0], expected)


def test_m0_predict_shape(dataset):
    model = build_model("M0", load_model_parameters())
    X = dataset.to_matrix()
    model.fit(X, dataset.Y)
    assert model.predict(X).shape == dataset.Y.shape


def test_m1_predict_proba_scores(dataset):
    model = build_model("M1", load_model_parameters())
    X = dataset.to_matrix()
    model.fit(X, dataset.Y)
    scores = predict_proba_scores(model, X)
    assert scores.shape == dataset.Y.shape
    assert np.isfinite(scores).all()


def test_m1_deterministic_with_seed(dataset):
    params = load_model_parameters()
    X = dataset.to_matrix()
    first = build_model("M1", params).fit(X, dataset.Y)
    second = build_model("M1", params).fit(X, dataset.Y)
    np.testing.assert_allclose(
        predict_proba_scores(first, X),
        predict_proba_scores(second, X),
    )


def test_m2_ensemble_three_chains(dataset):
    model = ClassifierChainEnsemble(seeds=[42, 43, 44])
    X = dataset.to_matrix()
    model.fit(X, dataset.Y)
    assert len(model.chains_) == 3
    scores = model.predict_proba(X)
    assert scores.shape == dataset.Y.shape


def test_m5_disabled_raises():
    with pytest.raises(ValueError, match="disabled"):
        build_model("M5", load_model_parameters())


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("MX", load_model_parameters())


def test_feature_matrix_experiment_a_no_indicator(dataset):
    matrix_a = feature_matrix(dataset, experiment="A")
    assert matrix_a.shape[0] == dataset.n_samples


def test_feature_matrix_experiment_b_adds_indicator(dataset):
    from spotify_intelligence.classification.training import train_imputation_values

    values = train_imputation_values(dataset)
    matrix_b = feature_matrix(dataset, experiment="B", imputation_values=values)
    assert matrix_b.shape[1] == feature_matrix(dataset, "A").shape[1] + 1


def test_serving_experiment_b_keeps_raw_time_signature_one_hot():
    from spotify_intelligence.classification.serving import build_recording_feature_row

    row = pd.Series(
        {
            "duration_ms": 180000.0,
            "key": 5.0,
            "mode": 1.0,
            "danceability": 0.5,
            "energy": 0.7,
            "loudness": -8.0,
            "speechiness": 0.1,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "liveness": 0.3,
            "valence": 0.6,
            "tempo": 120.0,
            "time_signature": 5.0,
            "audio_analysis_incomplete": True,
        }
    )
    imputation = {
        "tempo": 90.0,
        "danceability": 0.3,
        "speechiness": 0.05,
        "valence": 0.4,
        "time_signature": 4.0,
    }

    features_b = build_recording_feature_row(row, experiment="B", imputation_values=imputation)
    features_a = build_recording_feature_row(row, experiment="A")

    numeric_count = features_a.shape[1] - len(TIME_SIGNATURE_CATEGORIES)
    one_hot_b = features_b[0, numeric_count : numeric_count + len(TIME_SIGNATURE_CATEGORIES)]
    expected = np.array([float(cat == 5.0) for cat in TIME_SIGNATURE_CATEGORIES])
    np.testing.assert_array_equal(one_hot_b, expected)
    assert features_b.shape[1] == features_a.shape[1] + 1

    imputed_columns = [c for c in INCOMPLETE_AUDIO_COLUMNS if c != "time_signature"]
    for column in imputed_columns:
        assert features_b[0, _numeric_column_index(column)] == imputation[column]


def _numeric_column_index(column: str) -> int:
    base = {
        "danceability": 0,
        "energy": 1,
        "loudness": 2,
        "speechiness": 3,
        "acousticness": 4,
        "instrumentalness": 5,
        "liveness": 6,
        "valence": 7,
        "tempo": 8,
        "log_duration": 9,
        "key_sin": 10,
        "key_cos": 11,
        "mode": 12,
    }
    return base[column]
