"""Evaluate the final multiclass model once on the frozen test split (sección 24.4).

Usage:
    uv run python scripts/evaluate_final_multiclass_model.py --artifact-dir models/classifier/multiclass/<id> --use-test

The test split is used ONLY when ``--use-test`` is passed explicitly, and only
once, after the model is selected (sección 17.4, sección 17.5). Without the flag the script
refuses to run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from spotify_intelligence.classification.evaluation import (
    measure_prediction_latency,
    model_size_mb,
    save_metrics_report,
)
from spotify_intelligence.classification.multiclass import (
    expand_to_full_label_space,
    model_classes,
    predict_proba_scores,
)
from spotify_intelligence.classification.multiclass_evaluation import (
    dominant_genre_exploratory,
    evaluate_multiclass,
)
from spotify_intelligence.classification.training import (
    feature_matrix,
    prepare_base_dataset,
    prepare_multiclass_data,
    subset_dataset,
)
from spotify_intelligence.data.contracts import DataContractError
from spotify_intelligence.data.splits import load_splits
from spotify_intelligence.features.encoders import GenreLabelEncoder

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/metrics")


def multi_genre_test_rows(scaler):
    """Return scaled features and multi-label target for multi-genre test rows."""
    dataset = prepare_base_dataset(PROCESSED_DIR)
    split_map = load_splits(PROCESSED_DIR)
    test = subset_dataset(dataset, split_map["test"], experiment="A")
    mask = test.Y.sum(axis=1) > 1
    X = scaler.transform(feature_matrix(test, "A")[mask])
    Y = test.Y[mask]
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final multiclass model on test")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--use-test",
        action="store_true",
        help="Allow the one-time final evaluation on the frozen test split",
    )
    args = parser.parse_args()

    if not args.use_test:
        raise SystemExit(
            "Refusing to use the test split. Re-run with --use-test only for the "
            "one-time final evaluation after model selection (sección 17.4)."
        )

    artifact_dir = args.artifact_dir
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise DataContractError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    experiment = manifest["experiment"]
    model = joblib.load(artifact_dir / "model.joblib")
    classes = model_classes(model)
    encoder_payload = json.loads((artifact_dir / "genre_encoder.json").read_text(encoding="utf-8"))
    encoder = GenreLabelEncoder.load(encoder_payload)
    class_names = encoder.classes_

    # ``data.X_test`` is already scaled by ``prepare_multiclass_data``.
    data = prepare_multiclass_data(experiment=experiment, include_test=True)
    X_test = data.X_test
    y_test = data.y_test

    dense = predict_proba_scores(model, X_test)
    full = expand_to_full_label_space(dense, classes, len(class_names))
    metrics = evaluate_multiclass(y_test, full, class_names)
    metrics["latency_mean_ms"] = measure_prediction_latency(
        lambda X: predict_proba_scores(model, X), X_test
    )

    multi_X, multi_Y = multi_genre_test_rows(data.scaler)
    dense_mg = predict_proba_scores(model, multi_X)
    full_mg = expand_to_full_label_space(dense_mg, classes, len(class_names))
    exploratory = dominant_genre_exploratory(multi_Y, full_mg, class_names)

    report = {
        "experiment_id": manifest["experiment_id"],
        "model_name": manifest["model"],
        "model_id": manifest["model_id"],
        "experiment": experiment,
        "split_sha256": manifest["split_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "test_rows": int(len(X_test)),
        "model_size_mb": model_size_mb(artifact_dir),
        "metrics": metrics,
        "multi_genre_exploratory_test": exploratory,
        "test_used": True,
    }
    output_path = save_metrics_report(
        report, REPORT_DIR, name="multiclass_final_test_evaluation.json"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2))
    print("Accuracy (test):", round(metrics["accuracy"], 4))
    print("Macro F1 (test):", round(metrics["macro_f1"], 4))
    print("Hit@1 multigénero (test):", round(exploratory["hit_at_1"], 4))
    print("Report saved:", output_path)


if __name__ == "__main__":
    main()
