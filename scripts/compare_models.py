"""Compare multilabel classifiers M0..M4 on validation (AGENTS.md §16.6/§16.9).

Usage:
    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --models M0 M1

Trains each requested model with the frozen grouped split and reports the
validation metrics in ``reports/experiments/classifier_multilabel_comparison.json``.
The test split is never used; the threshold is tuned per model on validation.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from spotify_intelligence.classification.evaluation import (
    evaluate_multilabel,
    measure_prediction_latency,
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

REPORT_DIR = Path("reports/experiments")
ALL_MODELS = list(MODEL_IDS)  # M5 is disabled at build time


def run_comparison(models: list[str], experiment: str = "A") -> pd.DataFrame:
    """Train the requested models and return a validation comparison table."""
    model_params = load_model_parameters("configs/model_parameters.yaml")
    data = prepare_training_data(experiment=experiment)
    rows = []

    for model_id in models:
        model = build_model(model_id, model_params)
        start = time.perf_counter()
        model.fit(data.X_train, data.Y_train)
        training_seconds = time.perf_counter() - start

        val_scores = predict_proba_scores(model, data.X_val)
        threshold_result = tune_global_threshold(val_scores, data.Y_val)
        prediction = predict_with_threshold(
            val_scores, data.dataset.genre_encoder, threshold_result.best_threshold
        )
        metrics = evaluate_multilabel(data.Y_val, val_scores, prediction["labels"])
        metrics["latency_mean_ms"] = measure_prediction_latency(
            lambda X, fitted_model=model: predict_proba_scores(fitted_model, X),
            data.X_val,
        )

        row = {
            "model_id": model_id,
            "model_name": MODEL_IDS[model_id],
            "experiment": experiment,
            "training_seconds": round(training_seconds, 4),
            "best_threshold": threshold_result.best_threshold,
            "macro_f1": round(metrics["macro_f1"], 6),
            "micro_f1": round(metrics["micro_f1"], 6),
            "samples_f1": round(metrics["samples_f1"], 6),
            "precision_at_3": round(metrics["precision_at_3"], 6),
            "recall_at_5": round(metrics["recall_at_5"], 6),
            "hit_at_3": round(metrics["hit_at_3"], 6),
            "hit_at_5": round(metrics["hit_at_5"], 6),
            "hamming_loss": round(metrics["hamming_loss"], 6),
            "lrap": round(metrics["lrap"], 6) if metrics["lrap"] is not None else None,
            "latency_mean_ms": metrics["latency_mean_ms"],
            "train_rows": int(len(data.X_train)),
            "validation_rows": int(len(data.X_val)),
            "n_features": int(data.X_train.shape[1]),
        }
        rows.append(row)
        print(f"  {model_id} ({MODEL_IDS[model_id]}): samples_f1={row['samples_f1']}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multilabel classifiers")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=["M0", "M1"],
        help="Model ids to compare (default: M0 M1)",
    )
    parser.add_argument("--experiment", choices=["A", "B"], default="A")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_comparison(args.models, experiment=args.experiment)
    output_path = REPORT_DIR / "classifier_multilabel_comparison.json"

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": args.experiment,
        "dataset_sha256": compute_file_hash("data/raw/dataset.csv"),
        "models": summary.to_dict(orient="records"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
