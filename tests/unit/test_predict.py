import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import GENRES, make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.predict import (  # noqa: E402
    predict_with_threshold,
    top_k_genres,
)
from spotify_intelligence.features.encoders import GenreLabelEncoder  # noqa: E402


@pytest.fixture()
def encoder():
    return GenreLabelEncoder.from_genre_catalog(make_classification_fixture()["genre_catalog"])


def test_top_k_genres_returns_sorted_names(encoder):
    scores = np.array([[0.9, 0.1, 0.8, 0.2, 0.0, 0.3]])
    top = top_k_genres(scores, encoder, k=3)
    assert len(top[0]) == 3
    assert top[0][0] == GENRES[0]


def test_top_k_default_is_five(encoder):
    scores = np.zeros((1, 6))
    scores[0, :] = np.arange(6)
    top = top_k_genres(scores, encoder)
    assert len(top[0]) == 5


def test_predict_with_threshold_labels(encoder):
    scores = np.array([[0.95, 0.05, 0.05, 0.05, 0.05, 0.05]])
    result = predict_with_threshold(scores, encoder, threshold=0.5)
    assert result["labels"].sum() == 1
    assert not result["below_threshold"][0]


def test_predict_with_threshold_below_threshold_flag(encoder):
    scores = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])
    result = predict_with_threshold(scores, encoder, threshold=0.5)
    assert result["below_threshold"][0]
    assert result["labels"].sum() == 0
    assert len(result["top_k_genres"][0]) == 5


def test_top_k_matches_encoder_order(encoder):
    scores = np.zeros((1, 6))
    scores[0, 3] = 1.0
    top = top_k_genres(scores, encoder, k=1)
    assert top[0] == [GENRES[3]]
