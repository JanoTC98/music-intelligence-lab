import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.datasets import (  # noqa: E402
    MultilabelDataset,
    build_incomplete_mask,
    build_multiclass_dataset,
    build_multilabel_dataset,
)
from spotify_intelligence.classification.training import subset_dataset  # noqa: E402
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


def test_multiclass_dataset_preserves_incomplete_mask(fixture):
    dataset = build_multiclass_dataset(
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
        if flag and gid in set(dataset.recording_group_ids)
    }
    mask_groups = set(dataset.recording_group_ids[dataset.incomplete_mask].tolist())
    assert mask_groups == incomplete_groups
    assert len(dataset.incomplete_mask) == dataset.n_samples


def test_multiclass_experiment_a_drops_incomplete_rows(fixture):
    dataset = build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    split_map = fixture["split_map"]
    train_a = subset_dataset(dataset, split_map["train"], experiment="A")
    assert not train_a.incomplete_mask.any()

    dataset_groups_in_train = set(split_map["train"]) & set(dataset.recording_group_ids)
    incomplete_map = dict(
        zip(
            fixture["recordings"]["recording_group_id"],
            fixture["recordings"]["audio_analysis_incomplete"],
            strict=False,
        )
    )
    expected = len(dataset_groups_in_train) - sum(
        1 for gid in dataset_groups_in_train if incomplete_map[gid]
    )
    assert train_a.n_samples == expected


def test_multiclass_experiment_b_preserves_incomplete_rows(fixture):
    dataset = build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    split_map = fixture["split_map"]
    train_b = subset_dataset(dataset, split_map["train"], experiment="B")
    incomplete_map = dict(
        zip(
            fixture["recordings"]["recording_group_id"],
            fixture["recordings"]["audio_analysis_incomplete"],
            strict=False,
        )
    )
    expected = {
        gid
        for gid in set(split_map["train"]) & set(dataset.recording_group_ids)
        if incomplete_map[gid]
    }
    assert set(train_b.recording_group_ids[train_b.incomplete_mask].tolist()) == expected


def test_multiclass_dataset_aligns_mask_by_id_not_position(fixture):
    shuffled = fixture["recordings"].sample(frac=1.0, random_state=7).reset_index(drop=True)
    dataset = build_multiclass_dataset(
        shuffled,
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    reference = build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    assert list(dataset.recording_group_ids) == list(reference.recording_group_ids)
    np.testing.assert_array_equal(dataset.incomplete_mask, reference.incomplete_mask)


def test_build_incomplete_mask_alignment_by_id(fixture):
    feature_frame = add_engineered_features(fixture["recordings"])
    shuffled = fixture["recordings"].sample(frac=1.0, random_state=3).reset_index(drop=True)
    ordered = build_incomplete_mask(feature_frame, fixture["recordings"])
    disordered = build_incomplete_mask(feature_frame, shuffled)
    np.testing.assert_array_equal(ordered, disordered)
    assert ordered.dtype == bool
    assert len(ordered) == len(feature_frame)


def test_multilabel_dataset_rejects_mismatched_lengths(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    n = dataset.n_samples
    X = dataset.X.copy()
    Y = dataset.Y

    with pytest.raises(Exception, match="X length"):
        MultilabelDataset(
            X=X.iloc[: n - 1],
            Y=Y,
            genre_encoder=dataset.genre_encoder,
            recording_group_ids=dataset.recording_group_ids,
            feature_columns=dataset.feature_columns,
        )
    with pytest.raises(Exception, match="recording_group_ids length"):
        MultilabelDataset(
            X=X,
            Y=Y,
            genre_encoder=dataset.genre_encoder,
            recording_group_ids=dataset.recording_group_ids[: n - 1],
            feature_columns=dataset.feature_columns,
        )
    with pytest.raises(Exception, match="incomplete_mask length"):
        MultilabelDataset(
            X=X,
            Y=Y,
            genre_encoder=dataset.genre_encoder,
            recording_group_ids=dataset.recording_group_ids,
            feature_columns=dataset.feature_columns,
            incomplete_mask=np.ones(n - 1, dtype=bool),
        )


def test_multilabel_dataset_coerces_mask_to_bool(fixture):
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    rebuilt = MultilabelDataset(
        X=dataset.X,
        Y=dataset.Y,
        genre_encoder=dataset.genre_encoder,
        recording_group_ids=dataset.recording_group_ids,
        feature_columns=dataset.feature_columns,
        incomplete_mask=dataset.incomplete_mask.astype(int),
    )
    assert rebuilt.incomplete_mask.dtype == bool
    np.testing.assert_array_equal(rebuilt.incomplete_mask, dataset.incomplete_mask)
