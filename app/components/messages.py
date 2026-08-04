"""Small message helpers with consistent wording (AGENTS.md sección 18.4/sección 18.5)."""

from __future__ import annotations

from typing import Any

import streamlit as st

MISSING_ARTIFACT_HINT = (
    "El artefacto requerido no existe. Ejecute el script de construcción correspondiente."
)


def _format_decimal(value: float | None) -> str:
    if value is None:
        return "n/d"
    return f"{value:.3f}".replace(".", ",")


def render_validation_metrics(metrics: dict[str, Any] | None, *, kind: str) -> None:
    """Render the saved validation quality of a classifier bundle (sección 18.5).

    The numbers come from ``metrics_validation.json`` saved at training time and
    help users calibrate trust: with only 18 acoustic features the models are
    weak (sección 30). Validation metrics, never test metrics.
    """
    if not metrics:
        return
    if kind == "multilabel":
        parts = (
            f"macro-F1 {_format_decimal(metrics.get('macro_f1'))}",
            f"hit@5 {_format_decimal(metrics.get('hit_at_5'))}",
            f"LRAP {_format_decimal(metrics.get('lrap'))}",
        )
    elif kind == "multiclass":
        parts = (
            f"accuracy {_format_decimal(metrics.get('accuracy'))}",
            f"balanced {_format_decimal(metrics.get('balanced_accuracy'))}",
            f"top5 {_format_decimal(metrics.get('top5_accuracy'))}",
        )
    else:
        return
    st.caption("Calidad en validación (sin usar test): " + " · ".join(parts))


def missing_artifact(kind: str) -> None:
    """Render the standard message for a missing model artifact."""
    st.error(f"{kind}: {MISSING_ARTIFACT_HINT}")


def warn_note(text: str) -> None:
    """Render a warning with neutral wording."""
    st.warning(text)


def info_note(text: str) -> None:
    """Render an informational note."""
    st.info(text)


def error_note(text: str) -> None:
    """Render an error without exposing a stack trace (sección 18.5)."""
    st.error(text)


def empty_state(text: str) -> None:
    """Render an empty-state message."""
    st.info(text)
