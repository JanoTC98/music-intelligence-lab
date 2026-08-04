import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import (  # noqa: E402
    build_tiny_recommender,
    make_affinity_catalog,
    make_same_work_catalog,
)

from spotify_intelligence.recommenders.errors import ArtifactNotFoundError  # noqa: E402
from spotify_intelligence.recommenders.track_based import (  # noqa: E402
    RecommendationFilters,
    TrackRecommender,
)


@pytest.fixture()
def recommender(tmp_path):
    artifact_dir = build_tiny_recommender(tmp_path / "artifacts")
    return TrackRecommender(artifact_dir)


@pytest.fixture()
def recommender_same_work(tmp_path):
    artifact_dir = build_tiny_recommender(
        tmp_path / "artifacts_dup", catalog=make_same_work_catalog()
    )
    return TrackRecommender(artifact_dir)


@pytest.fixture()
def recommender_affinity(tmp_path):
    artifact_dir = build_tiny_recommender(
        tmp_path / "artifacts_affinity", catalog=make_affinity_catalog()
    )
    return TrackRecommender(artifact_dir)


def test_no_self_recommendation(recommender):
    for group_id in recommender.catalog_index["recording_group_id"]:
        results = recommender.recommend(group_id, top_n=5)
        assert group_id not in set(results["recording_group_id"])


def test_same_work_excluded_as_near_duplicate(recommender_same_work):
    seed = recommender_same_work.catalog_index.iloc[0]
    seed_name = str(seed["track_name"]).strip().casefold()
    seed_artists = str(seed["artists"]).strip().casefold()
    results = recommender_same_work.recommend(seed["recording_group_id"], top_n=20)
    same_work = results[
        results["track_name"].astype(str).str.strip().str.casefold().eq(seed_name)
        & results["artists"].astype(str).str.strip().str.casefold().eq(seed_artists)
    ]
    assert same_work.empty
    assert "g00b" not in set(results["recording_group_id"])


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
    catalog = recommender.catalog_index
    assert "audio_analysis_incomplete" in catalog.columns
    assert not catalog["audio_analysis_incomplete"].any()
    with pytest.raises(ArtifactNotFoundError):
        recommender.recommend("g07", top_n=5)


def test_partial_results_returned_when_expansion_exhausted(recommender):
    catalog = recommender.catalog_index
    recommender.manifest["retrieval"] = {
        "initial_candidate_floor": 2,
        "candidate_multiplier": 1,
        "expansion_steps": [4],
    }
    restrictive = RecommendationFilters(
        genres=["genre0"],
        duration_enabled=True,
        duration_min_seconds=1,
        duration_max_seconds=1000,
        different_artist=True,
    )
    results = recommender.recommend(
        catalog.iloc[0]["recording_group_id"], top_n=20, filters=restrictive
    )
    assert len(results) >= 1


def test_filter_duration(recommender):
    catalog = recommender.catalog_index
    group_id = catalog.iloc[0]["recording_group_id"]
    filters = RecommendationFilters(
        duration_enabled=True,
        duration_min_seconds=100,
        duration_max_seconds=400,
    )
    results = recommender.recommend(group_id, top_n=20, filters=filters)
    assert (results["duration_ms"] >= 100_000).all()
    assert (results["duration_ms"] <= 400_000).all()


def test_filter_popularity_min(recommender):
    catalog = recommender.catalog_index
    group_id = catalog.iloc[0]["recording_group_id"]
    filters = RecommendationFilters(popularity_min=23)
    results = recommender.recommend(group_id, top_n=20, filters=filters)
    assert (results["popularity_median"] >= 23).all()


def test_filter_different_artist(recommender):
    catalog = recommender.catalog_index
    seed = catalog.iloc[1]
    filters = RecommendationFilters(different_artist=True)
    results = recommender.recommend(seed["recording_group_id"], top_n=20, filters=filters)
    assert not (results["artists"] == seed["artists"]).any()


def test_from_config_respects_popularity_min_enabled():
    config = {
        "track_recommender": {
            "filters": {
                "explicit_default": "all",
                "duration_enabled_default": False,
                "duration_suggested_min_seconds": 60,
                "duration_suggested_max_seconds": 600,
                "different_artist_default": False,
                "popularity_min_enabled_default": True,
                "popularity_min_suggested": 30,
            }
        }
    }
    filters = RecommendationFilters.from_config(config)
    assert filters.popularity_min == 30


def test_from_config_disabled_popularity_filter():
    config = {
        "track_recommender": {
            "filters": {
                "explicit_default": "all",
                "duration_enabled_default": False,
                "duration_suggested_min_seconds": 60,
                "duration_suggested_max_seconds": 600,
                "different_artist_default": False,
                "popularity_min_enabled_default": False,
            }
        }
    }
    filters = RecommendationFilters.from_config(config)
    assert filters.popularity_min is None


def test_explanations_present_by_default(recommender):
    group_id = recommender.catalog_index.iloc[0]["recording_group_id"]
    results = recommender.recommend(group_id, top_n=5)
    assert "feature_differences" in results.columns
    assert len(results) > 0
    for row in results["feature_differences"]:
        assert isinstance(row, list)
        assert all("feature" in item and "difference" in item for item in row)


def test_explanations_optional(recommender):
    group_id = recommender.catalog_index.iloc[0]["recording_group_id"]
    results = recommender.recommend(group_id, top_n=5, include_explanations=False)
    assert "feature_differences" not in results.columns


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


def test_genre_affinity_disabled_keeps_similarity_order(recommender_affinity):
    group_id = recommender_affinity.catalog_index.iloc[0]["recording_group_id"]
    results = recommender_affinity.recommend(group_id, top_n=3, genre_affinity=False)
    assert results["recording_group_id"].tolist() == ["g01", "g02", "g04"]
    assert list(results["genres"].iloc[0]) == ["genre1"]


def test_genre_affinity_ranks_shared_genre_first(recommender_affinity):
    group_id = recommender_affinity.catalog_index.iloc[0]["recording_group_id"]
    results = recommender_affinity.recommend(group_id, top_n=3, genre_affinity=True)
    assert results["recording_group_id"].tolist() == ["g02", "g04", "g06"]
    for genres in results["genres"]:
        assert set(genres) == {"genre0"}


def test_genre_affinity_top_n_extended(recommender_affinity):
    group_id = recommender_affinity.catalog_index.iloc[0]["recording_group_id"]
    results = recommender_affinity.recommend(group_id, top_n=4, genre_affinity=True)
    assert results["recording_group_id"].tolist() == ["g02", "g04", "g06", "g01"]


def test_genre_affinity_keeps_pool_and_excludes_seed(recommender_affinity):
    catalog = recommender_affinity.catalog_index
    seed = catalog.iloc[0]
    pool = set(catalog["recording_group_id"]) - {seed["recording_group_id"]}
    results = recommender_affinity.recommend(
        seed["recording_group_id"], top_n=6, genre_affinity=True
    )
    assert set(results["recording_group_id"]) <= pool
    assert seed["recording_group_id"] not in set(results["recording_group_id"])


def test_genre_affinity_stable_across_calls(recommender_affinity):
    group_id = recommender_affinity.catalog_index.iloc[0]["recording_group_id"]
    first = recommender_affinity.recommend(group_id, top_n=4, genre_affinity=True)
    second = recommender_affinity.recommend(group_id, top_n=4, genre_affinity=True)
    assert list(first["recording_group_id"]) == list(second["recording_group_id"])
