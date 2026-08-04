"""Classifier artifact failure modes (AGENTS.md sección 18.4, sección 25.6).

A versioned bundle directory can exist on disk (manifests, thresholds and
encoders are tracked in git) while the joblib payload is absent, e.g. when only
part of the bundle was deployed or retraining produced a new timestamp. The
serving layer must fail with the controlled ``DataContractError`` so the app
renders the sección 18.4 message instead of leaking a raw ``FileNotFoundError``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from spotify_intelligence.classification.serving import (
    load_multiclass_serving,
    load_multilabel_serving,
    load_validation_metrics,
)
from spotify_intelligence.data.contracts import DataContractError


def _multilabel_bundle(tmp_path: Path, name: str, *, with_model: bool) -> Path:
    artifact_dir = tmp_path / "multilabel" / name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text(
        '{"experiment_id": "x", "model_id": "M1_B", "experiment": "B"}', encoding="utf-8"
    )
    (artifact_dir / "threshold.json").write_text('{"best_threshold": 0.85}', encoding="utf-8")
    if with_model:
        (artifact_dir / "model.joblib").write_bytes(b"payload")
        (artifact_dir / "scaler.joblib").write_bytes(b"payload")
    return artifact_dir


def test_multilabel_missing_joblib_raises_contract_error(tmp_path: Path) -> None:
    """Reproduces the production crash: dir present, model.joblib absent."""
    _multilabel_bundle(tmp_path, "20260803-0620_multilabel_M1_B", with_model=False)
    with pytest.raises(DataContractError):
        load_multilabel_serving("M1_B", models_dir=tmp_path)


def test_multilabel_unknown_model_key_raises_contract_error(tmp_path: Path) -> None:
    """A model key with no bundle raises the controlled error, not a crash."""
    with pytest.raises(DataContractError):
        load_multilabel_serving("NOPE", models_dir=tmp_path)


def test_multiclass_missing_joblib_raises_contract_error(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "multiclass" / "20260803-1619_multiclass_C1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text(
        '{"experiment_id": "x", "model_id": "C1", "experiment": "A"}', encoding="utf-8"
    )
    with pytest.raises(DataContractError):
        load_multiclass_serving("C1", models_dir=tmp_path)


def test_validation_metrics_none_when_file_missing(tmp_path: Path) -> None:
    """Missing metrics_validation.json yields None, not an error."""
    _multilabel_bundle(tmp_path, "20260803-0620_multilabel_M1_B", with_model=True)
    assert load_validation_metrics("multilabel", "M1_B", models_dir=tmp_path) is None
