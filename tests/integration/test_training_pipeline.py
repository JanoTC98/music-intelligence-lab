import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.datasets import build_multilabel_dataset  # noqa: E402
from spotify_intelligence.classification.multilabel import (  # noqa: E402
    build_model,
    load_model_parameters,
    predict_proba_scores,
)
from spotify_intelligence.classification.predict import predict_with_threshold  # noqa: E402
from spotify_intelligence.classification.thresholds import tune_global_threshold  # noqa: E402
from spotify_intelligence.classification.training import (  # noqa: E402
    subset_dataset,
)


def test_grouped_split_isolated_training_pipeline(tmp_path):
    fixture = make_classification_fixture()
    recordings = fixture["recordings"]
    recording_genres = fixture["recording_genres"]
    genre_catalog = fixture["genre_catalog"]
    split_map = fixture["split_map"]

    dataset = build_multilabel_dataset(recordings, recording_genres, genre_catalog)

    train_ds = subset_dataset(dataset, split_map["train"], experiment="A")
    val_ds = subset_dataset(dataset, split_map["validation"], experiment="A")

    X_train = train_ds.to_matrix()
    Y_train = train_ds.Y
    X_val = val_ds.to_matrix()
    Y_val = val_ds.Y

    assert not (set(train_ds.recording_group_ids) & set(val_ds.recording_group_ids))

    model = build_model("M1", load_model_parameters())
    model.fit(X_train, Y_train)
    scores = predict_proba_scores(model, X_val)

    assert scores.shape == Y_val.shape
    threshold = tune_global_threshold(scores, Y_val)
    prediction = predict_with_threshold(scores, val_ds.genre_encoder, threshold.best_threshold)

    assert prediction["labels"].shape == Y_val.shape
    assert len(prediction["top_k_genres"][0]) == 5


def test_prepare_base_dataset_synthetic_shape(tmp_path):
    fixture = make_classification_fixture()
    dataset = build_multilabel_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    assert dataset.n_labels == 6
    assert dataset.n_samples == len(fixture["recordings"])
    assert dataset.Y.dtype == np.int8
