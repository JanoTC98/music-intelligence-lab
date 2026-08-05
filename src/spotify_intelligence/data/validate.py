from __future__ import annotations

from typing import Any

import pandas as pd

from spotify_intelligence.data.contracts import (
    DataContractError,
    get_required_columns,
    get_validation_ranges,
)


def check_required_columns(df: pd.DataFrame) -> list[str]:
    required = get_required_columns()
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataContractError(f"Missing required columns: {missing}")
    return required


def validate_column_ranges(
    df: pd.DataFrame,
    ranges: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[tuple[int, Any]]]:
    if ranges is None:
        ranges = get_validation_ranges()

    violations: dict[str, list[tuple[int, Any]]] = {}

    for col, rules in ranges.items():
        if col not in df.columns:
            continue

        col_min = rules.get("min")
        col_max = rules.get("max")
        allowed_values = rules.get("allowed_values")

        mask = pd.Series(False, index=df.index)

        if allowed_values is not None:
            mask = ~df[col].isin(allowed_values)
        else:
            if col_min is not None:
                mask = mask | (df[col] < col_min)
            if col_max is not None:
                mask = mask | (df[col] > col_max)

        if mask.any():
            violators = df.index[mask].tolist()
            violations[col] = [(idx, df.loc[idx, col]) for idx in violators[:10]]

    return violations


def detect_incomplete_audio(
    df: pd.DataFrame,
    zero_columns: list[str] | None = None,
) -> pd.Series:
    if zero_columns is None:
        zero_columns = ["tempo", "danceability", "speechiness", "valence", "time_signature"]

    mask = pd.Series(True, index=df.index)
    for col in zero_columns:
        if col in df.columns:
            mask = mask & (df[col] == 0)

    return mask


def report_column_extremes(
    df: pd.DataFrame,
    ranges: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    """Report min/max for columns configured with ``report_extremes: True``.

    Used by the audit for unbounded numeric columns such as ``loudness``.
    """
    if ranges is None:
        ranges = get_validation_ranges()

    extremes: dict[str, dict[str, float]] = {}
    for col, rules in ranges.items():
        if not rules.get("report_extremes") or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        extremes[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return extremes
