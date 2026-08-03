"""Shared training-pipeline helpers for the multilabel module (§16.4/§16.5).

Centralizes data preparation so that the training script, the model
comparison script and the exploration notebook use identical transformations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from spotify_intelligence.classification.datasets import (
    MultilabelDataset,
    build_multilabel_dataset,
    load_processed_data,
)
from spotify_intelligence.data.splits import load_splits
from spotify_intelligence.features.audio_features import INCOMPLETE_AUDIO_COLUMNS

TIME_SIGNATURE_CATEGORIES = (0, 1, 3, 4, 5)


def prepare_base_dataset(processed_dir: str | Path = "data/processed") -> MultilabelDataset:
    """Load processed tables and build the full multilabel dataset."""
    tables = load_processed_data(processed_dir)
    return build_multilabel_dataset(
        tables["recordings"],
        tables["recording_genres"],
        tables["genre_catalog"],
    )


def split_map_from_dir(processed_dir: str | Path = "data/processed") -> dict[str, list[str]]:
    """Load the frozen grouped split mapping."""
    return load_splits(processed_dir)


def subset_dataset(
    dataset: MultilabelDataset,
    groups: list[str],
    experiment: str,
) -> MultilabelDataset:
    """Return the dataset rows for ``groups`` applying the experiment policy.

    Experiment A (baseline) drops ``audio_analysis_incomplete`` rows.
    Experiment B keeps them (imputation happens at the feature matrix stage).
    """
    sub, _ = dataset.split_rows(pd.Index(groups))
    if experiment == "A":
        mask = ~sub.incomplete_mask
        return MultilabelDataset(
            X=sub.X.loc[mask].reset_index(drop=True),
            Y=sub.Y[mask],
            genre_encoder=sub.genre_encoder,
            recording_group_ids=sub.recording_group_ids[mask],
            feature_columns=sub.feature_columns,
            incomplete_mask=sub.incomplete_mask[mask],
        )
    return sub


def train_imputation_values(train_dataset: MultilabelDataset) -> dict[str, float]:
    """Compute train-only medians for the incomplete-audio pattern (§16.5B)."""
    complete = train_dataset.X[~train_dataset.incomplete_mask]
    return {col: float(complete[col].median()) for col in INCOMPLETE_AUDIO_COLUMNS}


def impute_feature_frame(
    dataset: MultilabelDataset,
    values: dict[str, float],
) -> pd.DataFrame:
    """Return the feature frame with incomplete rows imputed from ``values``."""
    feature_frame = dataset.X.copy()
    for col in INCOMPLETE_AUDIO_COLUMNS:
        feature_frame.loc[dataset.incomplete_mask, col] = values[col]
    return feature_frame


def feature_matrix(
    dataset: MultilabelDataset,
    experiment: str,
    imputation_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the numeric feature matrix under experiment A or B.

    Columns: primary numeric features (excluding time_signature) + a fixed
    one-hot encoding of time_signature + (experiment B only) the incomplete
    audio indicator.
    """
    numeric = [c for c in dataset.feature_columns if c != "time_signature"]
    ts = dataset.X["time_signature"]
    one_hot = np.column_stack([(ts == cat).astype(float) for cat in TIME_SIGNATURE_CATEGORIES])

    if experiment == "B":
        if imputation_values is None:
            raise ValueError("experiment B requires train imputation values")
        feature_frame = impute_feature_frame(dataset, imputation_values)
        indicator = dataset.incomplete_mask.astype(int)
        return np.hstack(
            [
                feature_frame[numeric].to_numpy(dtype=float),
                one_hot,
                indicator.reshape(-1, 1),
            ]
        )
    return np.hstack([dataset.X[numeric].to_numpy(dtype=float), one_hot])


class TrainingData:
    """Prepared, scaled matrices for one experiment configuration."""

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        X_test: np.ndarray | None,
        Y_test: np.ndarray | None,
        scaler: StandardScaler,
        dataset: MultilabelDataset,
        experiment: str,
    ) -> None:
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.scaler = scaler
        self.dataset = dataset
        self.experiment = experiment


def prepare_training_data(
    *,
    experiment: str = "A",
    processed_dir: str | Path = "data/processed",
    include_test: bool = False,
) -> TrainingData:
    """Prepare scaled train/validation (/test) matrices for one experiment."""
    dataset = prepare_base_dataset(processed_dir)
    split_map = split_map_from_dir(processed_dir)

    train_dataset = subset_dataset(dataset, split_map["train"], experiment)
    val_dataset = subset_dataset(dataset, split_map["validation"], experiment)

    imputation_values = None
    if experiment == "B":
        full_train = subset_dataset(dataset, split_map["train"], "B")
        imputation_values = train_imputation_values(full_train)

    X_train_raw = feature_matrix(train_dataset, experiment, imputation_values)
    Y_train = train_dataset.Y
    X_val_raw = feature_matrix(val_dataset, experiment, imputation_values)
    Y_val = val_dataset.Y

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    X_test = None
    Y_test = None
    if include_test:
        test_dataset = subset_dataset(dataset, split_map["test"], experiment)
        X_test = scaler.transform(feature_matrix(test_dataset, experiment, imputation_values))
        Y_test = test_dataset.Y

    return TrainingData(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        scaler=scaler,
        dataset=dataset,
        experiment=experiment,
    )
