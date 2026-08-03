"""Evaluate the final multilabel model once on the frozen test split (§24.4).

Usage:
    uv run python scripts/evaluate_final_model.py --artifact-dir models/classifier/multilabel/<id>

The test split is used ONLY when ``--use-test`` is passed explicitly, and only
once, after the model is selected. Without the flag the script refuses to run,
guarding against accidental test leakage (§16.4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from spotify_intelligence.classification.evaluation import (
    evaluate_multilabel,
    measure_prediction_latency,
    model_size_mb,
    save_metrics_report,
)
from spotify_intelligence.classification.multilabel import predict_proba_scores
from spotify_intelligence.classification.predict import predict_with_threshold
from spotify_intelligence.classification.training import prepare_training_data
from spotify_intelligence.data.contracts import DataContractError
from spotify_intelligence.features.encoders import GenreLabelEncoder

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final multilabel model on test")
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
            "one-time final evaluation after model selection (§16.4)."
        )

    artifact_dir = args.artifact_dir
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise DataContractError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    experiment = manifest["experiment"]
    model = joblib.load(artifact_dir / "model.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    threshold = json.loads((artifact_dir / "threshold.json").read_text(encoding="utf-8"))[
        "best_threshold"
    ]
    encoder_payload = json.loads((artifact_dir / "genre_encoder.json").read_text(encoding="utf-8"))
    encoder = GenreLabelEncoder.load(encoder_payload)

    # Rebuild the frozen test split with the same pipeline as training.
    data = prepare_training_data(experiment=experiment, include_test=True)
    X_test = scaler.transform(data.X_test)
    Y_test = data.Y_test

    scores = predict_proba_scores(model, X_test)
    prediction = predict_with_threshold(scores, encoder, threshold)
    metrics = evaluate_multilabel(Y_test, scores, prediction["labels"])
    metrics["threshold"] = threshold
    metrics["latency_mean_ms"] = measure_prediction_latency(
        lambda X: predict_proba_scores(model, X), X_test
    )
    metrics["below_threshold_ratio"] = float(prediction["below_threshold"].mean())

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
        "test_used": True,
    }
    output_path = save_metrics_report(
        report, REPORT_DIR, name="multilabel_final_test_evaluation.json"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Report saved:", output_path)


if __name__ == "__main__":
    main()
