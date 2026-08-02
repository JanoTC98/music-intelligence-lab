from __future__ import annotations

from typing import Any


def explain_feature_differences(
    seed_features: dict[str, float],
    candidate_features: dict[str, float],
    std_scale: dict[str, float] | None = None,
    feature_labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Explain per-feature differences between a seed and a candidate.

    Values are expressed as absolute differences. When ``std_scale`` is provided,
    each difference is also reported in standard-deviation units.
    """
    labels = feature_labels or {}
    rows: list[dict[str, Any]] = []
    for feature, seed_value in seed_features.items():
        if feature not in candidate_features:
            continue
        candidate_value = candidate_features[feature]
        difference = float(candidate_value) - float(seed_value)
        row: dict[str, Any] = {
            "feature": feature,
            "label": labels.get(feature, feature),
            "seed_value": float(seed_value),
            "candidate_value": float(candidate_value),
            "difference": difference,
            "absolute_difference": abs(difference),
        }
        if std_scale and feature in std_scale and std_scale[feature]:
            row["difference_std"] = difference / std_scale[feature]
        rows.append(row)
    return rows


def format_bpm_difference(difference_std: float | None, std_bpm: float) -> str:
    """Format a tempo difference as BPM using a fixed standard deviation."""
    if difference_std is None:
        return ""
    return f"{(difference_std * std_bpm):.1f} BPM"
