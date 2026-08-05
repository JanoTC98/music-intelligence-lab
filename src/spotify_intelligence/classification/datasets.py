"""Classification dataset construction.

Builds a feature matrix ``X`` (one row per ``recording_group_id``) and a
multilabel target matrix ``Y`` (binary ``[n, 114]``) from the processed
catalog. The transformation is fully deterministic: feature engineering has no
fitted state, so train, validation and test receive identical treatment.

No scaler is applied at this stage because the configuration defines the primary
feature set verbatim; any scaling decision belongs to the experiment layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spotify_intelligence.data.contracts import DataContractError
from spotify_intelligence.features.audio_features import (
    PRIMARY_FEATURES,
    add_engineered_features,
)
from spotify_intelligence.features.encoders import (
    GenreLabelEncoder,
    TimeSignatureOneHot,
)

PROCESSED_DATA_FILES = {
    "recordings": "recordings.parquet",
    "recording_genres": "recording_genres.parquet",
    "genre_catalog": "genre_catalog.parquet",
}

FEATURE_COLUMNS: tuple[str, ...] = PRIMARY_FEATURES
TIME_SIGNATURE_COLUMN = "time_signature"


def load_processed_data(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the three processed tables used by the classifier."""
    base = Path(processed_dir)
    result: dict[str, pd.DataFrame] = {}
    for key, filename in PROCESSED_DATA_FILES.items():
        path = base / filename
        if not path.exists():
            raise DataContractError(f"Missing processed file: {path}")
        result[key] = pd.read_parquet(path)
    return result


class MultilabelDataset:
    """Feature matrix, binary target matrix and the canonical genre encoder."""

    def __init__(
        self,
        X: pd.DataFrame,
        Y: np.ndarray,
        genre_encoder: GenreLabelEncoder,
        recording_group_ids: pd.Index,
        feature_columns: list[str],
        incomplete_mask: np.ndarray | None = None,
    ) -> None:
        if len(Y) != len(X):
            raise DataContractError(f"Y length {len(Y)} does not match X length {len(X)}")
        if len(recording_group_ids) != len(X):
            raise DataContractError(
                f"recording_group_ids length {len(recording_group_ids)} "
                f"does not match X length {len(X)}"
            )
        if incomplete_mask is None:
            mask = np.zeros(len(X), dtype=bool)
        else:
            mask = np.asarray(incomplete_mask, dtype=bool)
            if len(mask) != len(X):
                raise DataContractError(
                    f"incomplete_mask length {len(mask)} does not match X length {len(X)}"
                )
        self.X = X
        self.Y = Y
        self.genre_encoder = genre_encoder
        self.recording_group_ids = recording_group_ids
        self.feature_columns = feature_columns
        self.incomplete_mask = mask

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_labels(self) -> int:
        return int(self.Y.shape[1])

    def to_matrix(self) -> np.ndarray:
        """Return the numeric feature matrix, one-hot included.

        ``time_signature`` is one-hot encoded using the fixed category set
        ``PRIMARY_TIME_SIGNATURES`` so shapes stay identical across splits.
        """
        numeric = [c for c in self.feature_columns if c != TIME_SIGNATURE_COLUMN]
        encoder = TimeSignatureOneHot()
        one_hot = encoder.transform(self.X[TIME_SIGNATURE_COLUMN])
        matrix = np.hstack([self.X[numeric].to_numpy(dtype=float), one_hot])
        return matrix

    def split_rows(
        self,
        recording_group_ids: pd.Index,
    ) -> tuple[MultilabelDataset, np.ndarray]:
        """Return a sub-dataset restricted to ``recording_group_ids``."""
        mask = np.asarray(self.recording_group_ids.isin(recording_group_ids), dtype=bool)
        sub = MultilabelDataset(
            X=self.X.loc[mask],
            Y=self.Y[mask],
            genre_encoder=self.genre_encoder,
            recording_group_ids=self.recording_group_ids[mask],
            feature_columns=self.feature_columns,
            incomplete_mask=self.incomplete_mask[mask],
        )
        return sub, mask


def build_incomplete_mask(
    feature_frame: pd.DataFrame,
    recordings: pd.DataFrame,
) -> np.ndarray:
    """Return a boolean mask aligned to ``feature_frame`` by ``recording_group_id``.

    A group is marked incomplete when ``recordings`` has an explicit
    ``audio_analysis_incomplete`` True for that group. Missing values become
    ``False`` only when the group has no explicit mark. Alignment is by
    identifier, not by row order, so ``recordings`` may be shuffled.
    """
    if "audio_analysis_incomplete" not in recordings.columns:
        return np.zeros(len(feature_frame), dtype=bool)
    incomplete_map = recordings.set_index("recording_group_id")["audio_analysis_incomplete"]
    return (
        feature_frame["recording_group_id"].map(incomplete_map).fillna(False).to_numpy(dtype=bool)
    )


def build_multilabel_dataset(
    recordings: pd.DataFrame,
    recording_genres: pd.DataFrame,
    genre_catalog: pd.DataFrame,
    *,
    features: tuple[str, ...] = FEATURE_COLUMNS,
) -> MultilabelDataset:
    """Build the multilabel dataset aligned to ``recordings`` rows.

    - Rows are the ``recording_group_id`` values of ``recordings`` in order.
    - ``X`` contains the primary audio features plus engineered columns.
    - ``Y`` is a binary ``[n, n_labels]`` matrix ordered by ``genre_catalog``.
    """
    encoder = GenreLabelEncoder.from_genre_catalog(genre_catalog)

    # Feature matrix aligned by recording_group_id.
    feature_frame = add_engineered_features(recordings)[["recording_group_id", *features]].copy()
    feature_frame = feature_frame.sort_values("recording_group_id").reset_index(drop=True)

    # Label matrix: pivot recording_genres -> [n, n_labels].
    label_order = encoder.classes_
    label_index = {name: idx for idx, name in enumerate(label_order)}
    label_frame = (
        recording_genres.copy()
        .assign(_col=recording_genres["track_genre"].map(label_index))
        .dropna(subset=["_col"])
    )
    label_frame["_row"] = label_frame["recording_group_id"].map(
        {gid: idx for idx, gid in enumerate(feature_frame["recording_group_id"])}
    )
    label_frame = label_frame.dropna(subset=["_row"]).astype({"_row": int, "_col": int})

    Y = np.zeros((len(feature_frame), len(label_order)), dtype=np.int8)
    Y[label_frame["_row"].to_numpy(), label_frame["_col"].to_numpy()] = 1

    X = feature_frame[list(features)].copy()
    feature_columns = list(features)
    incomplete_mask = build_incomplete_mask(feature_frame, recordings)
    return MultilabelDataset(
        X=X,
        Y=Y,
        genre_encoder=encoder,
        recording_group_ids=pd.Index(feature_frame["recording_group_id"].tolist()),
        feature_columns=feature_columns,
        incomplete_mask=incomplete_mask,
    )


def build_multiclass_dataset(
    recordings: pd.DataFrame,
    recording_genres: pd.DataFrame,
    genre_catalog: pd.DataFrame,
    *,
    features: tuple[str, ...] = FEATURE_COLUMNS,
) -> MultilabelDataset:
    """Build the single-label subset used by the multiclass module.

    Only recordings with exactly one genre are kept. The target ``Y`` is a
    one-hot row per recording; the encoder preserves the 114-genre ordering so
    the label space stays consistent with the multilabel module.
    """
    encoder = GenreLabelEncoder.from_genre_catalog(genre_catalog)

    counts = recording_genres.groupby("recording_group_id")["track_genre"].nunique()
    single_label_groups = counts[counts == 1].index
    single_genres = (
        recording_genres[recording_genres["recording_group_id"].isin(single_label_groups)]
        .drop_duplicates("recording_group_id")
        .set_index("recording_group_id")["track_genre"]
    )

    feature_frame = add_engineered_features(recordings)
    feature_frame = feature_frame[feature_frame["recording_group_id"].isin(single_label_groups)]
    feature_frame = feature_frame.sort_values("recording_group_id").reset_index(drop=True)

    label_index = {name: idx for idx, name in enumerate(encoder.classes_)}
    Y = np.zeros((len(feature_frame), len(label_index)), dtype=np.int8)
    for row, gid in enumerate(feature_frame["recording_group_id"]):
        genre = single_genres.loc[gid]
        if genre in label_index:
            Y[row, label_index[genre]] = 1

    X = feature_frame[list(features)].copy()
    incomplete_mask = build_incomplete_mask(feature_frame, recordings)
    return MultilabelDataset(
        X=X,
        Y=Y,
        genre_encoder=encoder,
        recording_group_ids=pd.Index(feature_frame["recording_group_id"].tolist()),
        feature_columns=list(features),
        incomplete_mask=incomplete_mask,
    )


def split_dataset_by_groups(
    dataset: MultilabelDataset,
    split_map: dict[str, list[str]],
    split_name: str,
) -> MultilabelDataset:
    """Return the sub-dataset for ``split_name`` using a group->split mapping."""
    groups = split_map.get(split_name, [])
    return dataset.split_rows(pd.Index(groups))[0]


def dataset_summary(dataset: MultilabelDataset) -> dict[str, Any]:
    """Return a small JSON-serializable summary of a dataset."""
    return {
        "n_samples": dataset.n_samples,
        "n_labels": dataset.n_labels,
        "feature_columns": dataset.feature_columns,
        "label_positives": int(dataset.Y.sum()),
        "mean_labels_per_sample": float(dataset.Y.sum(axis=1).mean()),
    }
