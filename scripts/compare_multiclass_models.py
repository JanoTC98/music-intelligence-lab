"""Compare multiclass classifiers C0..C3 on validation (AGENTS.md §17.3/§17.4).

Usage:
    uv run python scripts/compare_multiclass_models.py
    uv run python scripts/compare_multiclass_models.py --models C0 C1

Trains each requested model with the frozen grouped split, reports §17.4
metrics and the §17.5 exploratory evaluation on multi-genre validation rows in
``reports/experiments/classifier_multiclass_comparison.json``. The test split
is never used.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from spotify_intelligence.classification.evaluation import measure_prediction_latency
from spotify_intelligence.classification.multiclass import (
    MODEL_IDS,
    build_model,
    expand_to_full_label_space,
    load_model_parameters,
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
from spotify_intelligence.data.audit import compute_file_hash
from spotify_intelligence.data.splits import load_splits

REPORT_DIR = Path("reports/experiments")
PROCESSED_DIR = Path("data/processed")
ALL_MODELS = list(MODEL_IDS)  # C4 is disabled at build time


def multi_genre_validation_rows(scaler):
    """Return scaled features and multi-label target for multi-genre validation rows."""
    dataset = prepare_base_dataset(PROCESSED_DIR)
    split_map = load_splits(PROCESSED_DIR)
    val = subset_dataset(dataset, split_map["validation"], experiment="A")
    mask = val.Y.sum(axis=1) > 1
    X = scaler.transform(feature_matrix(val, "A")[mask])
    Y = val.Y[mask]
    return X, Y


def run_comparison(models: list[str]) -> pd.DataFrame:
    """Train the requested models and return a validation comparison table."""
    model_params = load_model_parameters("configs/model_parameters.yaml")
    data = prepare_multiclass_data(experiment="A")
    class_names = data.dataset.genre_encoder.classes_
    multi_X, multi_Y = multi_genre_validation_rows(data.scaler)
    rows = []

    for model_id in models:
        model = build_model(model_id, model_params)
        start = time.perf_counter()
        model.fit(data.X_train, data.y_train)
        training_seconds = time.perf_counter() - start

        dense = predict_proba_scores(model, data.X_val)
        full = expand_to_full_label_space(dense, model_classes(model), data.dataset.n_labels)
        metrics = evaluate_multiclass(data.y_val, full, class_names)
        metrics["latency_mean_ms"] = measure_prediction_latency(
            lambda X, m=model: predict_proba_scores(m, X),
            data.X_val,
        )

        dense_mg = predict_proba_scores(model, multi_X)
        full_mg = expand_to_full_label_space(dense_mg, model_classes(model), data.dataset.n_labels)
        exploratory = dominant_genre_exploratory(multi_Y, full_mg, class_names)

        row = {
            "model_id": model_id,
            "model_name": MODEL_IDS[model_id],
            "training_seconds": round(training_seconds, 4),
            "accuracy": round(metrics["accuracy"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
            "top3_accuracy": round(metrics["top3_accuracy"], 6),
            "top5_accuracy": round(metrics["top5_accuracy"], 6),
            "n_classes_seen": metrics["n_classes_seen"],
            "latency_mean_ms": metrics["latency_mean_ms"],
            "multi_genre_exploratory": exploratory,
            "train_rows": int(len(data.X_train)),
            "validation_rows": int(len(data.X_val)),
            "n_features": int(data.X_train.shape[1]),
        }
        rows.append(row)
        print(
            f"  {model_id} ({MODEL_IDS[model_id]}): "
            f"acc={row['accuracy']} macro_f1={row['macro_f1']} "
            f"hit@1={exploratory['hit_at_1']}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiclass classifiers")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=["C0", "C1", "C2", "C3"],
        help="Model ids to compare (default: C0 C1 C2 C3)",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_comparison(args.models)
    output_path = REPORT_DIR / "classifier_multiclass_comparison.json"

    split_hash = json.loads((PROCESSED_DIR / "splits_manifest.json").read_text(encoding="utf-8"))[
        "split_sha256"
    ]
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "task": "multiclass",
        "dataset_sha256": compute_file_hash("data/raw/dataset.csv"),
        "split_sha256": split_hash,
        "models": summary.to_dict(orient="records"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
