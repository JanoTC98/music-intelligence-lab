"""Audio feature engineering for the classification module.

Engineered features operate on the processed ``recordings`` catalog:

- ``log_duration = log1p(duration_ms)``
- ``key_sin = sin(2 * pi * key / 12)``
- ``key_cos = cos(2 * pi * key / 12)``

These transforms are deterministic and depend only on the row, so they can be
applied consistently to train, validation and test sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_FEATURES: tuple[str, ...] = ("log_duration", "key_sin", "key_cos")

PRIMARY_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    *ENGINEERED_FEATURES,
    "mode",
    "time_signature",
)

INCOMPLETE_AUDIO_COLUMNS: tuple[str, ...] = (
    "tempo",
    "danceability",
    "speechiness",
    "valence",
    "time_signature",
)


def log_duration(duration_ms: pd.Series | np.ndarray) -> np.ndarray:
    """Return ``log1p(duration_ms)`` as a float array."""
    return np.log1p(np.asarray(duration_ms, dtype=float))


def key_sin(key: pd.Series | np.ndarray) -> np.ndarray:
    """Return the sine of the pitch-class angle for each key."""
    return np.sin(2 * np.pi * np.asarray(key, dtype=float) / 12.0)


def key_cos(key: pd.Series | np.ndarray) -> np.ndarray:
    """Return the cosine of the pitch-class angle for each key."""
    return np.cos(2 * np.pi * np.asarray(key, dtype=float) / 12.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered feature columns added.

    Column creation is idempotent: existing engineered columns are overwritten.
    """
    result = df.copy()
    result["log_duration"] = log_duration(result["duration_ms"])
    result["key_sin"] = key_sin(result["key"])
    result["key_cos"] = key_cos(result["key"])
    return result


def incomplete_audio_indicator(df: pd.DataFrame) -> np.ndarray:
    """Return a binary indicator of the confirmed incomplete-analysis pattern.

    The pattern is ``tempo = danceability = speechiness = valence =
    time_signature = 0``. The stored ``audio_analysis_incomplete`` flag
    is preferred when available; this helper recomputes it from raw columns.
    """
    return (
        (df["tempo"] == 0)
        & (df["danceability"] == 0)
        & (df["speechiness"] == 0)
        & (df["valence"] == 0)
        & (df["time_signature"] == 0)
    ).to_numpy(dtype=bool)
