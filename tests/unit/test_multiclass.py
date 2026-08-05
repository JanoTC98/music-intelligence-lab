import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.datasets import build_multiclass_dataset  # noqa: E402
from spotify_intelligence.classification.multiclass import (  # noqa: E402
    FrequentClassBaseline,
    build_model,
    expand_to_full_label_space,
    load_model_parameters,
    model_classes,
    predict_proba_scores,
)
from spotify_intelligence.classification.training import feature_matrix  # noqa: E402


@pytest.fixture()
def dataset():
    fixture = make_classification_fixture()
    return build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )


@pytest.fixture()
def prepared(dataset):
    X = feature_matrix(dataset, experiment="A")
    y = dataset.Y.argmax(axis=1)
    return X, y


def test_build_model_all_supported_ids():
    params = load_model_parameters()
    for model_id in ("C0", "C1", "C2", "C3"):
        model = build_model(model_id, params)
        assert model is not None


def test_c4_disabled_raises():
    with pytest.raises(ValueError, match="disabled"):
        build_model("C4", load_model_parameters())


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("C9", load_model_parameters())


def test_c0_predicts_majority_class(prepared):
    X, y = prepared
    model = FrequentClassBaseline().fit(X, y)
    pred = model.predict(X)
    assert (pred == pred[0]).all()
    assert pred[0] in np.unique(y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))


def test_frequent_baseline_returns_original_non_consecutive_class_labels():
    X = np.zeros((5, 2))
    y = np.array([5, 5, 5, 9, 9])
    model = FrequentClassBaseline().fit(X, y)

    np.testing.assert_array_equal(
        model.predict(X),
        np.array([5, 5, 5, 5, 5]),
    )
    assert model.majority_class_ == 5
    assert model.majority_index_ == 0


def test_frequent_baseline_predict_matches_proba_argmax():
    X = np.zeros((4, 2))
    y = np.array([5, 5, 9, 5])
    model = FrequentClassBaseline().fit(X, y)
    predicted_from_proba = model.classes_[model.predict_proba(X).argmax(axis=1)]
    np.testing.assert_array_equal(model.predict(X), predicted_from_proba)


def test_frequent_baseline_rejects_empty_y():
    model = FrequentClassBaseline()
    with pytest.raises(ValueError, match="empty"):
        model.fit(np.zeros((0, 2)), np.array([], dtype=int))


def test_frequent_baseline_requires_fit_before_predict():
    model = FrequentClassBaseline()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.zeros((3, 2)))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(np.zeros((3, 2)))


def test_c0_classes_are_unique_sorted(prepared):
    X, y = prepared
    model = FrequentClassBaseline().fit(X, y)
    classes = model_classes(model)
    np.testing.assert_array_equal(classes, np.sort(np.unique(y)))


def test_c1_deterministic_with_seed(dataset):
    params = load_model_parameters()
    X = feature_matrix(dataset, experiment="A")
    y = dataset.Y.argmax(axis=1)
    first = build_model("C1", params).fit(X, y)
    second = build_model("C1", params).fit(X, y)
    np.testing.assert_allclose(
        predict_proba_scores(first, X),
        predict_proba_scores(second, X),
    )


@pytest.mark.parametrize("model_id", ["C2", "C3"])
def test_tree_models_predict_shapes(prepared, model_id):
    X, y = prepared
    model = build_model(model_id, load_model_parameters())
    model.fit(X, y)
    proba = predict_proba_scores(model, X)
    assert proba.shape == (len(X), len(model.classes_))
    assert len(model_classes(model)) == len(np.unique(y))


def test_expand_to_full_label_space():
    dense = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
    classes = np.array([0, 2, 5])
    full = expand_to_full_label_space(dense, classes, n_labels=6)
    assert full.shape == (2, 6)
    np.testing.assert_allclose(full[:, 5], dense[:, 2])
    np.testing.assert_allclose(full[:, 1], np.zeros(2))


def test_expand_requires_2d():
    with pytest.raises(ValueError, match="2D"):
        expand_to_full_label_space(np.zeros(3), np.array([0]), n_labels=6)
