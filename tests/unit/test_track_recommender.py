import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import build_tiny_recommender  # noqa: E402

from spotify_intelligence.recommenders.errors import ArtifactNotFoundError  # noqa: E402
from spotify_intelligence.recommenders.track_based import (  # noqa: E402
    RecommendationFilters,
    TrackRecommender,
)


@pytest.fixture()
def recommender(tmp_path):
    artifact_dir = build_tiny_recommender(tmp_path / "artifacts")
    return TrackRecommender(artifact_dir)


def test_no_self_recommendation(recommender):
    for group_id in recommender.catalog_index["recording_group_id"]:
        results = recommender.recommend(group_id, top_n=5)
        assert group_id not in set(results["recording_group_id"])


def test_no_duplicate_recording_groups(recommender):
    results = recommender.recommend(
        recommender.catalog_index.iloc[0]["recording_group_id"], top_n=5
    )
    assert results["recording_group_id"].is_unique


def test_filters_compliance_non_explicit(recommender):
    filters = RecommendationFilters(explicit="non_explicit")
    group_id = recommender.catalog_index.iloc[1]["recording_group_id"]
    results = recommender.recommend(group_id, top_n=5, filters=filters)
    assert not results["explicit"].any()


def test_filters_compliance_genre(recommender):
    catalog = recommender.catalog_index
    filters = RecommendationFilters(genres=["genre0"])
    group_id = catalog.iloc[0]["recording_group_id"]
    results = recommender.recommend(group_id, top_n=5, filters=filters)
    for genres in results["genres"]:
        assert set(genres) & {"genre0"}


def test_audio_incomplete_not_in_catalog(recommender):
    with pytest.raises(ArtifactNotFoundError):
        recommender.recommend("no-such-group", top_n=5)


def test_insufficient_results(recommender):
    catalog = recommender.catalog_index
    group_id = catalog.iloc[0]["recording_group_id"]
    results = recommender.recommend(group_id, top_n=len(catalog))
    assert len(results) <= len(catalog) - 1


def test_stable_order_on_ties(recommender):
    group_id = recommender.catalog_index.iloc[0]["recording_group_id"]
    first = recommender.recommend(group_id, top_n=5)
    second = recommender.recommend(group_id, top_n=5)
    assert list(first["recording_group_id"]) == list(second["recording_group_id"])
