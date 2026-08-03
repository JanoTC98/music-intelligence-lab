import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.datasets import (  # noqa: E402
    build_multiclass_dataset,
    build_multilabel_dataset,
)
from spotify_intelligence.features.audio_features import (  # noqa: E402
    add_engineered_features,
    key_cos,
    key_sin,
    log_duration,
)


@pytest.fixture()
def fixture():
    return make_classification_fixture()


def test_build_multilabel_dataset_shape(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    assert dataset.n_samples == len(fixture["recordings"])
    assert dataset.n_labels == 6
    assert dataset.Y.shape == (dataset.n_samples, 6)
    assert dataset.Y.sum(axis=1).min() >= 1


def test_dataset_has_no_forbidden_features(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    forbidden = {
        "track_id",
        "recording_group_id",
        "track_name",
        "artists",
        "album_name",
        "track_genre",
    }
    assert not (forbidden & set(dataset.feature_columns))


def test_dataset_incomplete_mask(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    incomplete_groups = {
        gid
        for gid, flag in zip(
            fixture["recordings"]["recording_group_id"],
            fixture["recordings"]["audio_analysis_incomplete"],
            strict=False,
        )
        if flag
    }
    mask_groups = set(dataset.recording_group_ids[dataset.incomplete_mask].tolist())
    assert mask_groups == incomplete_groups


def test_multilabel_target_aligned_to_recordings(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    assert list(dataset.recording_group_ids) == list(fixture["recordings"]["recording_group_id"])


def test_build_multiclass_dataset_single_label_only(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    mono = build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    assert mono.n_labels == 6
    assert (mono.Y.sum(axis=1) == 1).all()
    mono_groups = set(mono.recording_group_ids.tolist())
    multi_groups = set(dataset.recording_group_ids.tolist()) - mono_groups
    assert len(mono_groups) + len(multi_groups) == dataset.n_samples


def test_engineered_features_identity():
    df = pd.DataFrame({"duration_ms": [60_000], "key": [3]})
    out = add_engineered_features(df)
    assert out["log_duration"].iloc[0] == log_duration(60_000)
    assert out["key_sin"].iloc[0] == key_sin(3)
    assert out["key_cos"].iloc[0] == key_cos(3)


def test_to_matrix_shape_and_one_hot(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    matrix = dataset.to_matrix()
    n_numeric = len(dataset.feature_columns) - 1
    assert matrix.shape == (dataset.n_samples, n_numeric + 5)
