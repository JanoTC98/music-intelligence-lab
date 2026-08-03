import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import make_catalog_index  # noqa: E402

from spotify_intelligence.recommenders.errors import IncompatibleArtifactError  # noqa: E402
from spotify_intelligence.recommenders.scoring import (  # noqa: E402
    RECOMMENDER_FEATURES,
    build_eligible_recordings,
    cosine_similarity_from_distance,
    create_scaler,
    fit_scaler,
    prepare_catalog_matrix,
)


def test_cosine_similarity_from_distance():
    assert cosine_similarity_from_distance(0.0) == 1.0
    assert cosine_similarity_from_distance(0.5) == 0.5
    assert cosine_similarity_from_distance(1.0) == 0.0


def test_create_scaler_supported():
    from sklearn.preprocessing import RobustScaler, StandardScaler

    assert isinstance(create_scaler("standard"), StandardScaler)
    assert isinstance(create_scaler("robust"), RobustScaler)


def test_create_scaler_unsupported():
    with pytest.raises(IncompatibleArtifactError):
        create_scaler("minmax")


def test_build_eligible_recordings_excludes_incomplete():
    catalog = make_catalog_index(n=8)
    eligible = build_eligible_recordings(catalog, exclude_incomplete=True)
    assert not eligible["audio_analysis_incomplete"].any()
    assert len(eligible) == len(catalog) - 1


def test_build_eligible_recordings_keeps_incomplete_when_disabled():
    catalog = make_catalog_index(n=8)
    eligible = build_eligible_recordings(catalog, exclude_incomplete=False)
    assert len(eligible) == len(catalog)


def test_fit_scaler_and_prepare_catalog_matrix_shapes():
    catalog = make_catalog_index(n=8)
    matrix = catalog[list(RECOMMENDER_FEATURES)].to_numpy(dtype=float)

    scaler = fit_scaler(catalog, features=RECOMMENDER_FEATURES, scaler_name="standard")
    transformed = scaler.transform(matrix)
    assert transformed.shape == matrix.shape

    scaled, scaler2 = prepare_catalog_matrix(catalog, features=RECOMMENDER_FEATURES)
    assert scaled.shape == matrix.shape
    assert isinstance(scaled, np.ndarray)


def test_prepare_catalog_matrix_deterministic():
    catalog = make_catalog_index(n=8)
    scaled_a, _ = prepare_catalog_matrix(catalog, features=RECOMMENDER_FEATURES)
    scaled_b, _ = prepare_catalog_matrix(catalog, features=RECOMMENDER_FEATURES)
    np.testing.assert_allclose(scaled_a, scaled_b)
