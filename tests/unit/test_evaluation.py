import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import build_tiny_recommender  # noqa: E402

from spotify_intelligence.recommenders.evaluation import (  # noqa: E402
    evaluate_track_recommender,
    save_evaluation_report,
)
from spotify_intelligence.recommenders.track_based import TrackRecommender  # noqa: E402


@pytest.fixture()
def recommender(tmp_path):
    artifact_dir = build_tiny_recommender(tmp_path / "artifacts")
    return TrackRecommender(artifact_dir)


def test_evaluation_metrics_zero_violations(recommender):
    seed_rows = list(range(len(recommender.catalog_index)))
    report = evaluate_track_recommender(recommender, seed_rows, top_n=5, n_stability_samples=2)

    assert report["queries"] == len(recommender.catalog_index)
    assert report["self_recommendations"] == 0
    assert report["duplicate_groups"] == 0
    assert report["filter_violations"] == 0
    assert report["filter_compliance_pct"] == 100.0
    assert (
        report["mean_similarity"] is not None and -1.0 <= report["mean_similarity"] <= 1.0
    )  # similitud coseno, no una probabilidad (§14.10)
    assert report["latency_ms_p50"] >= 0


def test_evaluation_does_not_modify_artifacts(recommender, tmp_path):
    before = (tmp_path / "artifacts" / "catalog_matrix.npy").read_bytes()
    evaluate_track_recommender(recommender, [0], top_n=5, n_stability_samples=1)
    after = (tmp_path / "artifacts" / "catalog_matrix.npy").read_bytes()
    assert before == after


def test_save_evaluation_report_writes_json_and_csv(recommender, tmp_path):
    report = evaluate_track_recommender(recommender, [0], top_n=5, n_stability_samples=1)
    output_dir = tmp_path / "metrics"
    json_path = save_evaluation_report(report, output_dir)

    assert json_path.exists()
    assert (output_dir / "track_recommender_evaluation.csv").exists()
