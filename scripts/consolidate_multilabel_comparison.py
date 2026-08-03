"""Consolidate trained multilabel artifacts into the comparison report.

Reads every ``manifest.json`` under ``models/classifier/multilabel`` and builds
``reports/experiments/classifier_multilabel_comparison.json`` with the same
schema as ``scripts/compare_models.py`` but without re-training the models
(which would take tens of minutes for the tree ensembles).

Usage:
    uv run python scripts/consolidate_multilabel_comparison.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from spotify_intelligence.data.audit import compute_file_hash

MODELS_ROOT = Path("models/classifier/multilabel")
REPORT_PATH = Path("reports/experiments") / "classifier_multilabel_comparison.json"

_METRIC_KEYS = (
    "macro_f1",
    "micro_f1",
    "samples_f1",
    "precision_at_3",
    "recall_at_5",
    "hit_at_3",
    "hit_at_5",
    "hamming_loss",
    "lrap",
    "latency_mean_ms",
)


def _round(value, digits: int = 6):
    return round(value, digits) if isinstance(value, (int, float)) else value


def main() -> None:
    rows = []
    for artifact_dir in sorted(MODELS_ROOT.glob("*_multilabel_*_*")):
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation_metrics = manifest.get("validation_metrics", {})

        row = {
            "model_id": manifest["model_id"],
            "model_name": manifest["model"],
            "experiment": manifest["experiment"],
            "training_seconds": round(manifest["training_seconds"], 4),
            "best_threshold": validation_metrics.get("threshold"),
            "model_size_mb": manifest.get("model_size_mb"),
        }
        for key in _METRIC_KEYS:
            row[key] = _round(validation_metrics.get(key))
        row["train_rows"] = manifest.get("train_rows")
        row["validation_rows"] = manifest.get("validation_rows")
        row["n_features"] = manifest.get("n_features")
        row["artifact_path"] = str(artifact_dir)
        rows.append(row)

    rows.sort(key=lambda r: (r["experiment"], r["model_id"]))

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": compute_file_hash("data/raw/dataset.csv"),
        "models": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Report consolidated for {len(rows)} artifacts: {REPORT_PATH}")


if __name__ == "__main__":
    main()
