"""Artifact loading and single-recording inference for the app.

The Streamlit pages must never fit or train models. This module loads
the versioned classifier artifacts produced by the training scripts and applies
them to a single ``recording_group_id`` using the exact same feature recipe as
training ("misma transformación en entrenamiento e inferencia").

The feature row replicates ``classification.training.feature_matrix`` for one
row:

- experiment A: primary features (minus ``time_signature``) + fixed one-hot of
  ``time_signature`` -> 18 columns.
- experiment B: same as A plus the incomplete-audio indicator -> 19 columns,
  with incomplete rows imputed using train-only medians.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from spotify_intelligence.classification.multiclass import (
    expand_to_full_label_space,
    model_classes,
)
from spotify_intelligence.classification.multiclass import (
    predict_proba_scores as multiclass_scores,
)
from spotify_intelligence.classification.multilabel import (
    predict_proba_scores as multilabel_scores,
)
from spotify_intelligence.classification.predict import (
    DEFAULT_TOP_K,
    predict_with_threshold,
)
from spotify_intelligence.data.contracts import DataContractError
from spotify_intelligence.data.splits import load_splits
from spotify_intelligence.features.audio_features import (
    INCOMPLETE_AUDIO_COLUMNS,
    key_cos,
    key_sin,
    log_duration,
)
from spotify_intelligence.features.encoders import GenreLabelEncoder

TIME_SIGNATURE_CATEGORIES = (0, 1, 3, 4, 5)

MODELS_DIR = Path("models/classifier")


@dataclass(frozen=True)
class MultilabelServing:
    """Loaded multilabel artifacts for one experiment (M1_A or M1_B)."""

    experiment_id: str
    model_id: str
    experiment: str
    model: Any
    scaler: Any
    threshold: float
    encoder: GenreLabelEncoder
    manifest: dict[str, Any]


@dataclass(frozen=True)
class MulticlassServing:
    """Loaded multiclass artifacts for the final dominant-genre model (C1)."""

    experiment_id: str
    model_id: str
    experiment: str
    model: Any
    scaler: Any
    classes_: np.ndarray
    encoder: GenreLabelEncoder
    manifest: dict[str, Any]


def _experiment_dirs(kind: str, models_dir: str | Path) -> list[Path]:
    base = Path(models_dir) / kind
    if not base.exists():
        return []
    return sorted((path for path in base.iterdir() if path.is_dir()), reverse=True)


def discover_versions(kind: str, models_dir: str | Path = MODELS_DIR) -> list[str]:
    """Return experiment dir names for ``kind`` ordered newest first."""
    return [path.name for path in _experiment_dirs(kind, models_dir)]


def _find_artifact_dir(kind: str, model_key: str, models_dir: str | Path) -> Path:
    suffix = f"_{model_key}"
    for path in _experiment_dirs(kind, models_dir):
        if path.name.endswith(suffix):
            return path
    raise DataContractError(
        f"No {kind} artifact found for {model_key!r} under {Path(models_dir) / kind}"
    )


def _read_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise DataContractError(f"Missing manifest: {manifest_path}")
    import json

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _require_artifact_file(artifact_dir: Path, filename: str) -> Path:
    """Return an artifact path or raise a controlled ``DataContractError``.

    A bundle directory can exist (e.g. manifests tracked in git) while a
    joblib is missing on disk. Raising a typed error instead of a raw
    ``FileNotFoundError`` lets the app render the missing-artifact messaging.
    """
    path = artifact_dir / filename
    if not path.exists():
        raise DataContractError(f"Missing classifier artifact {filename} in {artifact_dir}")
    return path


def load_validation_metrics(
    kind: str,
    model_key: str,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any] | None:
    """Return the saved validation metrics for a serving bundle."""
    import json

    artifact_dir = _find_artifact_dir(kind, model_key, models_dir)
    path = artifact_dir / "metrics_validation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_encoder(artifact_dir: Path) -> GenreLabelEncoder:
    payload_path = artifact_dir / "genre_encoder.json"
    if not payload_path.exists():
        raise DataContractError(f"Missing genre encoder: {payload_path}")
    import json

    return GenreLabelEncoder.load(json.loads(payload_path.read_text(encoding="utf-8")))


def load_multilabel_serving(
    model_key: str = "M1_A",
    models_dir: str | Path = MODELS_DIR,
) -> MultilabelServing:
    """Load the final multilabel artifacts (M1 baseline by default)."""
    artifact_dir = _find_artifact_dir("multilabel", model_key, models_dir)
    manifest = _read_manifest(artifact_dir)
    threshold = json_helpers_load_threshold(artifact_dir)
    return MultilabelServing(
        experiment_id=manifest["experiment_id"],
        model_id=manifest["model_id"],
        experiment=str(manifest.get("experiment", "A")),
        model=joblib.load(_require_artifact_file(artifact_dir, "model.joblib")),
        scaler=joblib.load(_require_artifact_file(artifact_dir, "scaler.joblib")),
        threshold=threshold,
        encoder=_load_encoder(artifact_dir),
        manifest=manifest,
    )


def load_multiclass_serving(
    model_key: str = "C1",
    models_dir: str | Path = MODELS_DIR,
) -> MulticlassServing:
    """Load the final multiclass artifacts (C1 logistic by default)."""
    artifact_dir = _find_artifact_dir("multiclass", model_key, models_dir)
    manifest = _read_manifest(artifact_dir)
    model = joblib.load(_require_artifact_file(artifact_dir, "model.joblib"))
    return MulticlassServing(
        experiment_id=manifest["experiment_id"],
        model_id=manifest["model_id"],
        experiment=str(manifest.get("experiment", "A")),
        model=model,
        scaler=joblib.load(_require_artifact_file(artifact_dir, "scaler.joblib")),
        classes_=model_classes(model),
        encoder=_load_encoder(artifact_dir),
        manifest=manifest,
    )


def json_helpers_load_threshold(artifact_dir: Path) -> float:
    """Read the tuned threshold from ``threshold.json``."""
    import json

    path = artifact_dir / "threshold.json"
    if not path.exists():
        raise DataContractError(f"Missing threshold: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold = payload.get("best_threshold")
    if threshold is None:
        raise DataContractError(f"threshold.json has no best_threshold key: {path}")
    return float(threshold)


def load_recordings(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    """Load the canonical recording catalog."""
    return pd.read_parquet(Path(processed_dir) / "recordings.parquet")


def select_recording_row(
    recordings: pd.DataFrame,
    recording_group_id: str,
) -> pd.Series:
    """Return the catalog row for a recording group id (raises KeyError)."""
    frame = recordings.copy()
    frame = frame.set_index("recording_group_id")
    frame.index = frame.index.astype(str)
    if str(recording_group_id) not in frame.index:
        raise KeyError(f"recording_group_id not found: {recording_group_id}")
    return frame.loc[str(recording_group_id)]


def imputation_values_from_train(
    processed_dir: str | Path = "data/processed",
) -> dict[str, float]:
    """Return train-only medians for the incomplete-audio pattern.

    Mirrors ``classification.training.train_imputation_values`` but reads the
    processed recordings directly so the app does not build the full dataset.
    """
    processed = Path(processed_dir)
    recordings = pd.read_parquet(processed / "recordings.parquet")
    split_map = load_splits(processed)
    train_groups = set(split_map["train"])

    train = recordings[recordings["recording_group_id"].astype(str).isin(train_groups)]
    incomplete = train["audio_analysis_incomplete"].fillna(False).astype(bool)
    complete = train.loc[~incomplete, INCOMPLETE_AUDIO_COLUMNS]
    return {column: float(complete[column].median()) for column in INCOMPLETE_AUDIO_COLUMNS}


def build_recording_feature_row(
    recordings_row: pd.Series,
    *,
    experiment: str = "A",
    imputation_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the single-row feature vector using the training recipe.

    ``recordings_row`` must provide ``duration_ms``, ``key`` and every
    ``INCOMPLETE_AUDIO_COLUMNS`` + ``time_signature`` column. For experiment B,
    ``imputation_values`` must be supplied (train-only medians).

    The ``time_signature`` one-hot always uses the raw stored value, mirroring
    ``classification.training.feature_matrix``; imputation never rewrites the
    one-hot, so training and serving stay consistent.
    """
    row = recordings_row.copy()
    time_signature = float(row["time_signature"])

    if experiment == "B":
        if imputation_values is None:
            raise ValueError("experiment B requires train imputation values")
        incomplete = bool(row.get("audio_analysis_incomplete", False))
        if incomplete:
            for column in INCOMPLETE_AUDIO_COLUMNS:
                if column == "time_signature":
                    continue
                row[column] = imputation_values[column]

    numeric = [
        float(row[column])
        for column in (
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        )
    ]
    numeric.append(float(log_duration(np.asarray([float(row["duration_ms"])]))[0]))
    numeric.append(float(key_sin(np.asarray([float(row["key"])]))[0]))
    numeric.append(float(key_cos(np.asarray([float(row["key"])]))[0]))
    numeric.append(float(row["mode"]))

    one_hot = [float(time_signature == category) for category in TIME_SIGNATURE_CATEGORIES]

    features = [*numeric, *one_hot]
    if experiment == "B":
        indicator = float(bool(row.get("audio_analysis_incomplete", False)))
        features.append(indicator)
    elif experiment != "A":
        raise ValueError(f"Unknown experiment: {experiment!r}")

    return np.asarray([features], dtype=float)


def predict_multilabel_recording(
    serving: MultilabelServing,
    recordings_row: pd.Series,
    *,
    processed_dir: str | Path = "data/processed",
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run the multilabel model on one recording and return labeled output."""
    imputation_values = None
    if serving.experiment == "B":
        imputation_values = imputation_values_from_train(processed_dir)

    row = build_recording_feature_row(
        recordings_row,
        experiment=serving.experiment,
        imputation_values=imputation_values,
    )
    scaled = serving.scaler.transform(row)
    scores = multilabel_scores(serving.model, scaled)
    prediction = predict_with_threshold(scores, serving.encoder, serving.threshold, k=k)

    order = np.argsort(-scores[0])[:k]
    top_k = [
        {"genre": serving.encoder.classes_[int(idx)], "score": float(scores[0][int(idx)])}
        for idx in order
    ]
    return {
        "scores": scores[0],
        "top_k": top_k,
        "below_threshold": bool(prediction["below_threshold"][0]),
        "threshold": serving.threshold,
    }


def predict_multiclass_recording(
    serving: MulticlassServing,
    recordings_row: pd.Series,
    *,
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run the dominant-genre model on one recording and return Top-k output.

    The dense model scores are expanded to the full 114-label space
    and ranked; the scores remain uncalibrated.
    """
    row = build_recording_feature_row(recordings_row, experiment="A")
    scaled = serving.scaler.transform(row)
    dense = multiclass_scores(serving.model, scaled)
    full = expand_to_full_label_space(dense, serving.classes_, serving.encoder.n_labels)[0]

    order = np.argsort(-full)
    top_k = [
        {"genre": serving.encoder.classes_[int(idx)], "score": float(full[int(idx)])}
        for idx in order[:k]
    ]
    return {"top_k": top_k, "scores": full}
