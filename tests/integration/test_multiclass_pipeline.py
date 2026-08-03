import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from classifier_helpers import make_classification_fixture  # noqa: E402

from spotify_intelligence.classification.datasets import build_multiclass_dataset  # noqa: E402
from spotify_intelligence.classification.multiclass import (  # noqa: E402
    build_model,
    expand_to_full_label_space,
    load_model_parameters,
    model_classes,
    predict_proba_scores,
)
from spotify_intelligence.classification.multiclass_evaluation import (  # noqa: E402
    evaluate_multiclass,
)
from spotify_intelligence.classification.predict import top_k_genres  # noqa: E402
from spotify_intelligence.classification.training import (  # noqa: E402
    feature_matrix,
    subset_dataset,
)
from spotify_intelligence.features.encoders import GenreLabelEncoder  # noqa: E402


def test_multiclass_grouped_split_isolation():
    fixture = make_classification_fixture()
    dataset = build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    split_map = fixture["split_map"]
    train_ds = subset_dataset(dataset, split_map["train"], experiment="A")
    val_ds = subset_dataset(dataset, split_map["validation"], experiment="A")
    assert not (set(train_ds.recording_group_ids) & set(val_ds.recording_group_ids))


def test_no_forbidden_features():
    fixture = make_classification_fixture()
    dataset = build_multiclass_dataset(
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


def test_c1_train_evaluate_roundtrip(tmp_path):
    fixture = make_classification_fixture()
    dataset = build_multiclass_dataset(
        fixture["recordings"],
        fixture["recording_genres"],
        fixture["genre_catalog"],
    )
    split_map = fixture["split_map"]
    train_ds = subset_dataset(dataset, split_map["train"], experiment="A")
    val_ds = subset_dataset(dataset, split_map["validation"], experiment="A")

    X_train = feature_matrix(train_ds, experiment="A")
    y_train = train_ds.Y.argmax(axis=1)
    X_val = feature_matrix(val_ds, experiment="A")
    y_val = val_ds.Y.argmax(axis=1)

    model = build_model("C1", load_model_parameters())
    model.fit(X_train, y_train)

    dense = predict_proba_scores(model, X_val)
    full = expand_to_full_label_space(dense, model_classes(model), n_labels=dataset.n_labels)
    assert full.shape == (len(X_val), dataset.n_labels)

    class_names = dataset.genre_encoder.classes_
    metrics = evaluate_multiclass(y_val, full, class_names)
    assert metrics["accuracy"] >= 0.0

    topk = top_k_genres(full, dataset.genre_encoder, k=3)
    assert len(topk) == len(X_val)
    assert len(topk[0]) == 3

    out = tmp_path / "artifacts"
    out.mkdir()
    joblib.dump(model, out / "model.joblib")
    with open(out / "genre_encoder.json", "w", encoding="utf-8") as f:
        json.dump(dataset.genre_encoder.save(), f)
    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump(model_classes(model).tolist(), f)

    loaded = joblib.load(out / "model.joblib")
    encoder = GenreLabelEncoder.load(json.loads((out / "genre_encoder.json").read_text()))
    classes = np.asarray(json.loads((out / "classes.json").read_text()))
    dense2 = predict_proba_scores(loaded, X_val)
    full2 = expand_to_full_label_space(dense2, classes, n_labels=encoder.n_labels)
    np.testing.assert_allclose(full2, full)
