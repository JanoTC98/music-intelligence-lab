import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from recommender_helpers import make_catalog_index  # noqa: E402

from spotify_intelligence.recommenders import experiments as exp  # noqa: E402


@pytest.fixture()
def recordings():
    return make_catalog_index(n=8)


def test_exp_id_names():
    assert exp.EXP_ID_NAMES[("standard", "cosine")] == "R1"
    assert exp.EXP_ID_NAMES[("robust", "cosine")] == "R2"
    assert exp.EXP_ID_NAMES[("standard", "euclidean")] == "R3"
    assert exp.EXP_ID_NAMES[("robust", "euclidean")] == "R4"


def test_build_experiment_index_shapes(recordings):
    scaled, neighbors = exp.build_experiment_index(recordings)
    assert scaled.shape[0] == len(recordings)
    assert hasattr(neighbors, "kneighbors")


def test_recommend_with_index_excludes_seed(recordings):
    scaled, neighbors = exp.build_experiment_index(recordings)
    rows, dists = exp.recommend_with_index(scaled, neighbors, seed_row=0, top_n=5)
    assert 0 not in rows
    assert len(rows) <= 5
    assert len(rows) == len(dists)


def test_run_experiment_metrics(recordings):
    result = exp.run_experiment(
        recordings,
        seed_rows=[0, 1, 2],
        scaler_name="standard",
        metric="cosine",
        top_n=5,
        candidate_floor=100,
    )
    assert result["id"] == "R1"
    assert result["self_recommendations"] == 0
    assert result["duplicate_groups"] == 0
    assert result["comparable_similarity"] is True
    assert 0.0 <= result["catalog_coverage"] <= 1.0
    assert result["latency_mean_ms"] > 0


def test_run_experiment_euclidean_not_comparable(recordings):
    result = exp.run_experiment(
        recordings,
        seed_rows=[0],
        scaler_name="standard",
        metric="euclidean",
        top_n=5,
        candidate_floor=100,
    )
    assert result["id"] == "R3"
    assert result["comparable_similarity"] is False


def test_run_all_experiments_returns_four_rows(recordings):
    summary = exp.run_all_experiments(recordings, seed_rows=[0, 1, 2], top_n=5)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary["id"]) == ["R1", "R2", "R3", "R4"]


def test_save_experiment_report(tmp_path, recordings):
    summary = exp.run_all_experiments(recordings, seed_rows=[0], top_n=5)
    path = exp.save_experiment_report(summary, output_dir=tmp_path)
    assert path.exists()
    assert path.name == "track_recommender_r1_r4_comparison.json"
