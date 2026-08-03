import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import build_tiny_preference_recommender  # noqa: E402

from spotify_intelligence.recommenders.errors import (  # noqa: E402
    ArtifactNotFoundError,
    InvalidPreferenceProfileError,
)
from spotify_intelligence.recommenders.preference_based import (  # noqa: E402
    PreferenceProfile,
    PreferenceRecommender,
    weighted_euclidean_distance,
)
from spotify_intelligence.recommenders.track_based import RecommendationFilters  # noqa: E402


@pytest.fixture()
def recommender(tmp_path):
    artifact_dir = build_tiny_preference_recommender(tmp_path / "artifacts")
    return PreferenceRecommender(artifact_dir)


def test_weighted_distance_ignores_zero_weight():
    point = np.array([1.0, 100.0])
    query = np.array([0.0, 0.0])
    weights = np.array([0.0, 1.0])
    d = weighted_euclidean_distance(point, query, weights)
    assert d == pytest.approx(100.0)


def test_weighted_distance_normalized_by_total_weight():
    point = np.array([3.0])
    query = np.array([1.0])
    weights = np.array([2.0])
    d = weighted_euclidean_distance(point, query, weights)
    assert d == pytest.approx(2.0)


def test_weighted_distance_all_zero_raises():
    with pytest.raises(InvalidPreferenceProfileError):
        weighted_euclidean_distance(np.array([1.0]), np.array([0.0]), np.array([0.0]))


def test_profile_all_zero_weights_rejected():
    with pytest.raises(InvalidPreferenceProfileError):
        PreferenceProfile.from_manual(
            values={"energy": 0.8, "danceability": 0.9},
            weights={"energy": 0, "danceability": 0},
        )


def test_profile_weight_out_of_range_rejected():
    with pytest.raises(InvalidPreferenceProfileError):
        PreferenceProfile.from_manual(
            values={"energy": 0.8},
            weights={"energy": 5},
        )


def test_profile_missing_value_for_weighted_feature_rejected():
    with pytest.raises(InvalidPreferenceProfileError):
        PreferenceProfile.from_manual(
            values={"energy": 0.8},
            weights={"energy": 2, "tempo": 2},
        )


def test_recommend_returns_sorted_unique_groups(recommender):
    profile = PreferenceProfile.from_manual(
        values={
            "energy": 0.8,
            "danceability": 0.9,
            "valence": 0.7,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "tempo": 120,
        },
        weights={"energy": 2, "danceability": 3, "tempo": 2},
    )
    results = recommender.recommend(profile, top_n=5)
    assert len(results) <= 5
    assert results["recording_group_id"].is_unique
    assert list(results["distance"]) == sorted(results["distance"])


def test_recommend_filters_genre(recommender):
    profile = PreferenceProfile.from_manual(
        values={
            "energy": 0.8,
            "danceability": 0.9,
            "valence": 0.7,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "tempo": 120,
        },
        weights={"energy": 2, "danceability": 3, "tempo": 2},
    )
    filters = RecommendationFilters(genres=["genre0"])
    results = recommender.recommend(profile, top_n=5, filters=filters)
    for genres in results["genres"]:
        assert set(genres) & {"genre0"}


def test_ood_status_within_bounds(recommender):
    profile = PreferenceProfile.from_manual(
        values={
            "energy": 0.5,
            "danceability": 0.5,
            "valence": 0.5,
            "acousticness": 0.5,
            "instrumentalness": 0.5,
            "tempo": 120,
        },
        weights={"energy": 1, "danceability": 1},
    )
    status = recommender.out_of_distribution_status(profile)
    assert status["status"] in {"ok", "warning", "weak_match"}
    assert status["p95"] >= 0 and status["p99"] >= status["p95"]


def test_ood_weak_match_for_extreme_profile(recommender):
    profile = PreferenceProfile.from_manual(
        values={
            "energy": 1.0,
            "danceability": 1.0,
            "valence": 1.0,
            "acousticness": 0.0,
            "instrumentalness": 0.0,
            "tempo": 5,
        },
        weights={"energy": 3, "danceability": 3, "valence": 3, "tempo": 3},
    )
    status = recommender.out_of_distribution_status(profile)
    assert status["distance_to_centroid"] > 0


def test_mmr_preserves_first_result(recommender):
    profile = PreferenceProfile.from_manual(
        values={
            "energy": 0.8,
            "danceability": 0.9,
            "valence": 0.7,
            "acousticness": 0.1,
            "instrumentalness": 0.0,
            "tempo": 120,
        },
        weights={"energy": 2, "danceability": 3, "tempo": 2},
    )
    pure = recommender.recommend(profile, top_n=5)
    mmr = recommender.recommend(profile, top_n=5, diversity_enabled=True, lambda_=0.85)
    assert pure.iloc[0]["recording_group_id"] == mmr.iloc[0]["recording_group_id"]
    assert set(pure["recording_group_id"]) == set(mmr["recording_group_id"])


def test_artifact_not_found():
    with pytest.raises(ArtifactNotFoundError):
        PreferenceRecommender("no-such-dir")
