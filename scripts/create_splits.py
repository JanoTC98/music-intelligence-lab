"""Create and freeze the grouped classification splits (sección 16.4).

Usage:
    uv run python scripts/create_splits.py

Output (data/processed/):
    splits.parquet, splits_manifest.json

The test split is frozen: ``splits_manifest.json["split_sha256"]`` anchors the
exact assignment, so later runs must reproduce the same hash.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from spotify_intelligence.classification.datasets import (
    build_multilabel_dataset,
    load_processed_data,
)
from spotify_intelligence.config import load_yaml_config
from spotify_intelligence.data.audit import compute_file_hash
from spotify_intelligence.data.splits import (
    create_grouped_splits,
    save_splits,
    split_sha256,
)

PROCESSED_DIR = Path("data/processed")


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
    config = load_yaml_config("configs/classifier_features.yaml")
    splits_config = config["splits"]

    tables = load_processed_data(PROCESSED_DIR)
    dataset = build_multilabel_dataset(
        tables["recordings"],
        tables["recording_genres"],
        tables["genre_catalog"],
    )

    split_map = create_grouped_splits(
        dataset.recording_group_ids,
        dataset.Y,
        train_fraction=splits_config["train_fraction"],
        validation_fraction=splits_config["validation_fraction"],
        test_fraction=splits_config["test_fraction"],
        n_candidates=splits_config["candidate_split_count"],
        random_state=splits_config["random_state"],
    )

    split_hash = split_sha256(split_map)
    manifest = {
        "config_sha256": compute_file_hash("configs/classifier_features.yaml"),
        "dataset_sha256": compute_file_hash("data/raw/dataset.csv"),
        "random_state": splits_config["random_state"],
        "candidate_split_count": splits_config["candidate_split_count"],
        "freeze_test": splits_config["freeze_test"],
        "git_commit": _git_commit(),
    }
    save_splits(split_map, PROCESSED_DIR, split_hash=split_hash, manifest=manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
