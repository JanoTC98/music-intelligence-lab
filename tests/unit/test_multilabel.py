import sys
from pathlib import Path

import numpy as np
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
from spotify_intelligence.classification.training import feature_matrix  # noqa: E402


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
