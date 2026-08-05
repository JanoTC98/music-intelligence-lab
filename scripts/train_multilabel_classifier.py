"""Train one multilabel classifier.

Usage:
    uv run python scripts/train_multilabel_classifier.py --model M1 [--experiment A]
    uv run python scripts/train_multilabel_classifier.py --model M0 --experiment B

- ``experiment`` A (default): excludes ``audio_analysis_incomplete`` rows.
- ``experiment`` B: imputes incomplete rows with train statistics and adds a
  binary indicator. Imputation statistics are computed on train only.

The test split is never used here; the tuned threshold uses validation only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib

from spotify_intelligence.classification.evaluation import (
    build_experiment_manifest,
    evaluate_multilabel,
    measure_prediction_latency,
    model_size_mb,
    save_manifest,
    save_metrics_report,
)
from spotify_intelligence.classification.multilabel import (
    MODEL_IDS,
    build_model,
    load_model_parameters,
    predict_proba_scores,
)
from spotify_intelligence.classification.predict import predict_with_threshold
from spotify_intelligence.classification.thresholds import tune_global_threshold
from spotify_intelligence.classification.training import prepare_training_data
from spotify_intelligence.data.audit import compute_file_hash

PROCESSED_DIR = Path("data/processed")
MODELS_ROOT = Path("models/classifier/multilabel")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multilabel classifier")
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS))
    parser.add_argument("--experiment", choices=["A", "B"], default="A")
    parser.add_argument("--output-root", default=str(MODELS_ROOT))
    args = parser.parse_args()

    model_params = load_model_parameters("configs/model_parameters.yaml")
    split_hash = json.loads((PROCESSED_DIR / "splits_manifest.json").read_text(encoding="utf-8"))[
        "split_sha256"
    ]

    data = prepare_training_data(experiment=args.experiment)
    model = build_model(args.model, model_params)

    started_at = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()
    model.fit(data.X_train, data.Y_train)
    training_seconds = time.perf_counter() - start_time

    val_scores = predict_proba_scores(model, data.X_val)
    threshold_result = tune_global_threshold(val_scores, data.Y_val)
    val_prediction = predict_with_threshold(
        val_scores, data.dataset.genre_encoder, threshold_result.best_threshold
    )
    metrics = evaluate_multilabel(data.Y_val, val_scores, val_prediction["labels"])
    metrics["threshold"] = threshold_result.best_threshold
    metrics["threshold_curve"] = threshold_result.curve
    metrics["latency_mean_ms"] = measure_prediction_latency(
        lambda X: predict_proba_scores(model, X), data.X_val
    )

    experiment_id = f"{datetime.now(UTC):%Y%m%d-%H%M}_multilabel_{args.model}_{args.experiment}"
    out_dir = Path(args.output_root) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(data.scaler, out_dir / "scaler.joblib")
    with open(out_dir / "genre_encoder.json", "w", encoding="utf-8") as f:
        json.dump(data.dataset.genre_encoder.save(), f, ensure_ascii=False)
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_threshold": threshold_result.best_threshold,
                "best_score": threshold_result.best_score,
                "optimize_metric": threshold_result.optimize_metric,
                "experiment": args.experiment,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest = build_experiment_manifest(
        task="multilabel",
        model_name=MODEL_IDS[args.model],
        dataset_sha256=compute_file_hash("data/raw/dataset.csv"),
        split_sha256=split_hash,
        config_sha256=compute_file_hash("configs/model_parameters.yaml"),
        git_commit=_git_commit(),
        random_state=model_params["random_state"],
        test_used=False,
        training_seconds=training_seconds,
        artifact_path=str(out_dir),
        started_at_utc=started_at,
        experiment_id=experiment_id,
        extra={
            "model_id": args.model,
            "experiment": args.experiment,
            "feature_config_sha256": compute_file_hash("configs/classifier_features.yaml"),
            "model_size_mb": model_size_mb(out_dir),
            "train_rows": int(len(data.X_train)),
            "validation_rows": int(len(data.X_val)),
            "n_features": int(data.X_train.shape[1]),
            "validation_metrics": {k: v for k, v in metrics.items() if k != "threshold_curve"},
        },
    )
    save_manifest(manifest, out_dir)
    save_metrics_report(metrics, out_dir, name="metrics_validation.json")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Best threshold:", threshold_result.best_threshold)
    print("Samples F1 (validation):", round(metrics["samples_f1"], 4))


if __name__ == "__main__":
    main()
