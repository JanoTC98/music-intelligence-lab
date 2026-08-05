"""Grouped train/validation/test splits for the classification modules.

Splits are performed on ``recording_group_id`` so a full group belongs to a
single set. Multiple candidate splits are generated with
shuffled seeds, empty intersections are verified, and the candidate with the
smallest label-prevalence deviation is selected. The test split is frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from spotify_intelligence.data.contracts import DataContractError

SPLIT_NAMES = ("train", "validation", "test")


def generate_split_candidates(
    recording_group_ids: pd.Index,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    n_candidates: int = 50,
    random_state: int = 42,
) -> list[dict[str, list[str]]]:
    """Generate ``n_candidates`` grouped splits as ``{split: [group ids]}``.

    For each candidate, ``test`` is carved first (``test_fraction``), then
    ``validation`` is carved from the remainder at ``validation / (1 - test)``.
    Groups never cross split boundaries.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if not np.isclose(sum(fractions), 1.0):
        raise DataContractError(f"Split fractions must sum to 1, got {sum(fractions)}")

    groups = recording_group_ids.to_numpy()
    group_idx = pd.Index(groups)
    candidates: list[dict[str, list[str]]] = []

    for candidate in range(n_candidates):
        seed = random_state + candidate
        # Carve test (15%) first, grouped.
        test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_val_idx, test_idx = next(test_splitter.split(np.zeros(len(groups)), groups=groups))
        train_val = group_idx[train_val_idx].tolist()
        test = group_idx[test_idx].tolist()

        # Carve validation from the remaining 85%.
        remainder_fraction = validation_fraction / (1.0 - test_fraction)
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=remainder_fraction, random_state=seed + 1000
        )
        train_idx, val_idx = next(
            val_splitter.split(np.zeros(len(train_val)), groups=np.asarray(train_val))
        )
        train = np.asarray(train_val)[train_idx].tolist()
        validation = np.asarray(train_val)[val_idx].tolist()

        candidates.append(
            {
                "train": list(map(str, train)),
                "validation": list(map(str, validation)),
                "test": list(map(str, test)),
            }
        )
    return candidates


def verify_disjoint_splits(split_map: dict[str, list[str]]) -> None:
    """Raise ``DataContractError`` if any group appears in two splits."""
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(split_map[left]) & set(split_map[right])
        if overlap:
            raise DataContractError(
                f"Leaked groups between {left} and {right}: {len(overlap)} shared"
            )


def label_prevalence_deviation(
    split_map: dict[str, list[str]],
    y_labels: np.ndarray,
    recording_group_ids: pd.Index,
) -> float:
    """Return the mean absolute deviation of label prevalence across splits.

    For each split, per-label prevalence is compared against the overall
    prevalence; deviations are averaged over labels and splits.
    """
    group_to_row = {gid: idx for idx, gid in enumerate(recording_group_ids)}
    overall = np.asarray(y_labels, dtype=float).mean(axis=0)

    total_deviation = 0.0
    n_splits = 0
    for split in SPLIT_NAMES:
        rows = [group_to_row[gid] for gid in split_map[split] if gid in group_to_row]
        if not rows:
            continue
        split_prevalence = np.asarray(y_labels)[rows].mean(axis=0)
        total_deviation += float(np.mean(np.abs(split_prevalence - overall)))
        n_splits += 1
    return total_deviation / max(n_splits, 1)


def select_best_split(
    candidates: list[dict[str, list[str]]],
    y_labels: np.ndarray,
    recording_group_ids: pd.Index,
) -> dict[str, list[str]]:
    """Return the candidate split with the smallest label-prevalence deviation."""
    if not candidates:
        raise DataContractError("No split candidates were generated")
    best = min(
        candidates,
        key=lambda split_map: label_prevalence_deviation(split_map, y_labels, recording_group_ids),
    )
    return best


def create_grouped_splits(
    recording_group_ids: pd.Index,
    y_labels: np.ndarray,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    n_candidates: int = 50,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Generate, validate and select the frozen grouped split."""
    candidates = generate_split_candidates(
        recording_group_ids,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        n_candidates=n_candidates,
        random_state=random_state,
    )
    best = select_best_split(candidates, y_labels, recording_group_ids)
    verify_disjoint_splits(best)
    return best


def split_sha256(split_map: dict[str, list[str]]) -> str:
    """Return a stable hash of the split assignment (freeze test)."""
    payload = json.dumps(
        {split: sorted(split_map[split]) for split in SPLIT_NAMES},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_splits(
    split_map: dict[str, list[str]],
    output_dir: str | Path,
    *,
    split_hash: str,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Write ``splits.parquet`` and ``splits_manifest.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"recording_group_id": group, "split": split}
        for split in SPLIT_NAMES
        for group in split_map[split]
    ]
    df = pd.DataFrame(rows)
    df = df.sort_values("recording_group_id").reset_index(drop=True)
    parquet_path = output_dir / "splits.parquet"
    df.to_parquet(parquet_path, index=False)

    manifest_data = {
        "split_sha256": split_hash,
        "train_count": len(split_map["train"]),
        "validation_count": len(split_map["validation"]),
        "test_count": len(split_map["test"]),
        "total_count": sum(len(split_map[s]) for s in SPLIT_NAMES),
    }
    if manifest:
        manifest_data.update(manifest)
    with open(output_dir / "splits_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    return parquet_path


def load_splits(processed_dir: str | Path) -> dict[str, list[str]]:
    """Load ``splits.parquet`` back into a ``{split: [group ids]}`` mapping."""
    path = Path(processed_dir) / "splits.parquet"
    if not path.exists():
        raise DataContractError(f"Missing splits file: {path}")
    df = pd.read_parquet(path)
    return {
        split: df.loc[df["split"] == split, "recording_group_id"].astype(str).tolist()
        for split in SPLIT_NAMES
    }
