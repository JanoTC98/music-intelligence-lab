"""Encoders for classification targets and categorical audio features.

- ``GenreLabelEncoder`` maps the 114 genre names to a stable binary vector.
  The order follows ``genre_catalog.parquet`` (genre_id order), which is the
  versioned authority for label ordering (§25.3 "las etiquetas mantienen orden
  estable").
- ``TimeSignatureOneHot`` one-hot encodes ``time_signature`` using categories
  learned at fit time (train only) and ignores unseen categories at transform.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from spotify_intelligence.data.contracts import DataContractError

PRIMARY_TIME_SIGNATURES: tuple[int, ...] = (0, 1, 3, 4, 5)


class GenreLabelEncoder:
    """Encode genre names to stable binary vectors ordered by genre_id."""

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self._index: dict[str, int] = {}

    @classmethod
    def from_genre_catalog(cls, genre_catalog: pd.DataFrame) -> GenreLabelEncoder:
        """Build the encoder from the versioned ``genre_catalog.parquet``.

        The catalog must contain a ``track_genre`` column. Rows are kept in
        their stored order (genre_id order) to guarantee label stability.
        """
        encoder = cls()
        if "track_genre" not in genre_catalog.columns:
            raise DataContractError("genre_catalog must contain a track_genre column")
        encoder.classes_ = [str(value) for value in genre_catalog["track_genre"].tolist()]
        if len(encoder.classes_) != len(set(encoder.classes_)):
            raise DataContractError("genre_catalog contains duplicate genre names")
        encoder._index = {name: idx for idx, name in enumerate(encoder.classes_)}
        return encoder

    @property
    def n_labels(self) -> int:
        """Return the number of label columns."""
        return len(self.classes_)

    def fit(self, genres: Iterable[str]) -> GenreLabelEncoder:
        """Build the encoder from an iterable of genre names.

        Order is the first-seen order; prefer ``from_genre_catalog`` for the
        canonical versioned order.
        """
        self.classes_ = []
        for genre in genres:
            name = str(genre)
            if name not in self._index:
                self._index[name] = len(self.classes_)
                self.classes_.append(name)
        return self

    def transform(self, genres: Iterable[str]) -> np.ndarray:
        """Return a single binary row for a collection of genre names."""
        row = np.zeros(self.n_labels, dtype=np.int8)
        for genre in genres:
            idx = self._index.get(str(genre))
            if idx is not None:
                row[idx] = 1
        return row

    def transform_batch(self, genre_lists: Iterable[Iterable[str]]) -> np.ndarray:
        """Return a binary matrix ``[n_rows, n_labels]`` for many rows."""
        return np.vstack([self.transform(genres) for genres in genre_lists])

    def to_labels(self, binary_row: np.ndarray) -> list[str]:
        """Return genre names for the True positions of a binary row."""
        return [self.classes_[idx] for idx in np.flatnonzero(np.asarray(binary_row))]

    def top_k(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the top-k (genre, score) pairs from a row of scores."""
        order = np.argsort(-np.asarray(scores))
        return [(self.classes_[idx], float(scores[idx])) for idx in order[:k]]

    def save(self) -> dict[str, Any]:
        return {"classes": self.classes_}

    @classmethod
    def load(cls, payload: dict[str, Any]) -> GenreLabelEncoder:
        encoder = cls()
        encoder.classes_ = list(payload["classes"])
        encoder._index = {name: idx for idx, name in enumerate(encoder.classes_)}
        return encoder


class TimeSignatureOneHot:
    """One-hot encoder for ``time_signature`` with train-only categories.

    The default category set is the versioned observation ``{0, 1, 3, 4, 5}``.
    Categories can be re-learned at fit; unseen values at transform produce an
    all-zero vector so training and inference shapes always match.
    """

    def __init__(self, categories: Iterable[int] | None = None) -> None:
        if categories is None:
            categories = PRIMARY_TIME_SIGNATURES
        self.categories_: list[int] = sorted(int(c) for c in categories)

    def fit(self, series: pd.Series) -> TimeSignatureOneHot:
        """Learn categories from ``series`` (expected to be a train split)."""
        self.categories_ = sorted(int(v) for v in series.dropna().unique())
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        """Return the one-hot matrix ``[n_rows, len(categories)]``."""
        values = np.asarray(series, dtype=float)
        matrix = np.zeros((len(values), len(self.categories_)), dtype=np.float64)
        for col, category in enumerate(self.categories_):
            matrix[:, col] = values == category
        return matrix

    def save(self) -> dict[str, Any]:
        return {"categories": self.categories_}

    @classmethod
    def load(cls, payload: dict[str, Any]) -> TimeSignatureOneHot:
        return cls(categories=payload["categories"])


def sklearn_time_signature_encoder(
    categories: Iterable[int] | None = None,
) -> OneHotEncoder:
    """Return a scikit-learn one-hot encoder for ``time_signature``.

    Uses ``handle_unknown="ignore"`` so validation/test rows with a category
    absent from training map to an all-zero vector (shape stays consistent).
    """
    if categories is None:
        categories = PRIMARY_TIME_SIGNATURES
    encoder = OneHotEncoder(
        categories=[sorted(int(c) for c in categories)],
        sparse_output=False,
        handle_unknown="ignore",
        dtype=np.float64,
    )
    return encoder
