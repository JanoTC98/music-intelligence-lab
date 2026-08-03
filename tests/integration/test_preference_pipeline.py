import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import build_tiny_preference_recommender  # noqa: E402

from spotify_intelligence.recommenders.preference_based import (  # noqa: E402
    PreferenceProfile,
    PreferenceRecommender,
)


def test_build_then_recommend_preferences(tmp_path):
    artifact_dir = build_tiny_preference_recommender(tmp_path / "artifacts")
    recommender = PreferenceRecommender(artifact_dir)

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
    results = recommender.recommend(profile, top_n=3)

    assert len(results) <= 3
    assert results["recording_group_id"].is_unique

    status = recommender.out_of_distribution_status(profile)
    assert status["status"] in {"ok", "warning", "weak_match"}
