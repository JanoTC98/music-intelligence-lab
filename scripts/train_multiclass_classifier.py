"""Train one multiclass (dominant genre) classifier (AGENTS.md sección 17, sección 24.4).

Usage:
    uv run python scripts/train_multiclass_classifier.py --model C1
    uv run python scripts/train_multiclass_classifier.py --model C2

- Models C0..C3 are supported; C4 (XGBoost) is disabled in this project.
- The single-label subset (mono-genre recordings) is used under experiment A
  (``audio_analysis_incomplete`` rows excluded).
- No threshold is tuned: the predicted class is ``argmax`` of the scores.
- The test split is never used here; the final evaluation runs once with
  ``scripts/evaluate_final_multiclass_model.py --use-test``.
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
    measure_prediction_latency,
    model_size_mb,
    save_manifest,
    save_metrics_report,
)
from spotify_intelligence.classification.multiclass import (
    MODEL_IDS,
    build_model,
    expand_to_full_label_space,
    load_model_parameters,
    model_classes,
    predict_proba_scores,
)
from spotify_intelligence.classification.multiclass_evaluation import evaluate_multiclass
from spotify_intelligence.classification.training import prepare_multiclass_data
from spotify_intelligence.data.audit import compute_file_hash

PROCESSED_DIR = Path("data/processed")
MODELS_ROOT = Path("models/classifier/multiclass")


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
    parser = argparse.ArgumentParser(description="Train a multiclass classifier")
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS))
    parser.add_argument("--experiment", choices=["A"], default="A")
    parser.add_argument("--output-root", default=str(MODELS_ROOT))
    args = parser.parse_args()

    model_params = load_model_parameters("configs/model_parameters.yaml")
    split_hash = json.loads((PROCESSED_DIR / "splits_manifest.json").read_text(encoding="utf-8"))[
        "split_sha256"
    ]

    data = prepare_multiclass_data(experiment=args.experiment)
    model = build_model(args.model, model_params)

    started_at = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    training_seconds = time.perf_counter() - start_time

    dense_scores = predict_proba_scores(model, data.X_val)
    full_scores = expand_to_full_label_space(
        dense_scores,
        model_classes(model),
        data.dataset.n_labels,
    )
    metrics = evaluate_multiclass(data.y_val, full_scores, data.dataset.genre_encoder.classes_)
    metrics["latency_mean_ms"] = measure_prediction_latency(
        lambda X: predict_proba_scores(model, X), data.X_val
    )

    experiment_id = f"{datetime.now(UTC):%Y%m%d-%H%M}_multiclass_{args.model}"
    out_dir = Path(args.output_root) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(data.scaler, out_dir / "scaler.joblib")
    with open(out_dir / "genre_encoder.json", "w", encoding="utf-8") as f:
        json.dump(data.dataset.genre_encoder.save(), f, ensure_ascii=False)
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(model_classes(model).tolist(), f)

    manifest = build_experiment_manifest(
        task="multiclass",
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
            "n_classes": int(len(model_classes(model))),
            "validation_metrics": metrics,
        },
    )
    save_manifest(manifest, out_dir)
    save_metrics_report(metrics, out_dir, name="metrics_validation.json")

    print("Artifact:", out_dir)
    print("Experiment id:", experiment_id)
    print("Accuracy (validation):", round(metrics["accuracy"], 4))
    print("Macro F1 (validation):", round(metrics["macro_f1"], 4))
    print("Top-5 accuracy (validation):", round(metrics["top5_accuracy"], 4))


if __name__ == "__main__":
    main()
