import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import build_tiny_recommender  # noqa: E402

from spotify_intelligence.recommenders.track_based import TrackRecommender  # noqa: E402


def test_build_then_recommend(tmp_path):
    artifact_dir = build_tiny_recommender(tmp_path / "artifacts")
    recommender = TrackRecommender(artifact_dir)
    catalog = recommender.catalog_index

    assert len(catalog) == 7  # g07 es audio_analysis_incomplete y se excluye
    group_id = catalog.iloc[0]["recording_group_id"]
    results = recommender.recommend(group_id, top_n=3)

    assert len(results) <= 3
    assert group_id not in set(results["recording_group_id"])
    assert results["recording_group_id"].is_unique
