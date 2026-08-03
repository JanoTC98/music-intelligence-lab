import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import (  # noqa: E402
    GENRES,
    make_classification_fixture,
)

from spotify_intelligence.data.contracts import DataContractError  # noqa: E402
from spotify_intelligence.data.splits import (  # noqa: E402
    create_grouped_splits,
    label_prevalence_deviation,
    save_splits,
    split_sha256,
    verify_disjoint_splits,
)
from spotify_intelligence.features.encoders import GenreLabelEncoder  # noqa: E402


@pytest.fixture()
def fixture():
    return make_classification_fixture()


def test_generated_split_disjoint_and_exhaustive(fixture):
    recordings = fixture["recordings"]
    encoder = GenreLabelEncoder.from_genre_catalog(fixture["genre_catalog"])
    y_labels = encoder.transform_batch(
        fixture["recording_genres"].groupby("recording_group_id")["track_genre"].apply(list)
    )

    split_map = create_grouped_splits(
        recordings["recording_group_id"],
        y_labels,
        n_candidates=5,
    )
    verify_disjoint_splits(split_map)

    all_groups = set(recordings["recording_group_id"])
    assigned = set(split_map["train"]) | set(split_map["validation"]) | set(split_map["test"])
    assert assigned == all_groups


def test_verify_disjoint_raises_on_overlap(fixture):
    split_map = fixture["split_map"]
    with pytest.raises(DataContractError):
        verify_disjoint_splits(
            {
                "train": split_map["train"],
                "validation": split_map["train"],
                "test": split_map["test"],
            }
        )


def test_label_prevalence_deviation_non_negative(fixture):
    split_map = fixture["split_map"]
    encoder = GenreLabelEncoder.from_genre_catalog(fixture["genre_catalog"])
    y_labels = encoder.transform_batch(
        fixture["recording_genres"].groupby("recording_group_id")["track_genre"].apply(list)
    )
    deviation = label_prevalence_deviation(
        split_map,
        y_labels,
        fixture["recordings"]["recording_group_id"],
    )
    assert deviation >= 0.0


def test_split_sha256_is_stable(fixture):
    split_map = fixture["split_map"]
    assert split_sha256(split_map) == split_sha256(split_map)


def test_split_sha256_changes_with_assignment(fixture):
    split_map = fixture["split_map"]
    altered = {
        "train": split_map["train"][1:],
        "validation": split_map["validation"],
        "test": [split_map["train"][0], *split_map["test"]],
    }
    assert split_sha256(altered) != split_sha256(split_map)


def test_save_and_load_splits_roundtrip(tmp_path, fixture):
    split_map = fixture["split_map"]
    save_splits(split_map, tmp_path, split_hash="abc123")
    from spotify_intelligence.data.splits import load_splits

    loaded = load_splits(tmp_path)
    assert set(loaded["train"]) == set(split_map["train"])
    assert set(loaded["validation"]) == set(split_map["validation"])
    assert set(loaded["test"]) == set(split_map["test"])


def test_genre_encoder_stable_order(fixture):
    encoder = GenreLabelEncoder.from_genre_catalog(fixture["genre_catalog"])
    assert encoder.classes_ == GENRES
