"""Small message helpers with consistent wording (AGENTS.md §18.4/§18.5)."""

from __future__ import annotations

import streamlit as st

MISSING_ARTIFACT_HINT = (
    "El artefacto requerido no existe. Ejecute el script de construcción correspondiente."
)


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
    """Render an error without exposing a stack trace (§18.5)."""
    st.error(text)


def empty_state(text: str) -> None:
    """Render an empty-state message."""
    st.info(text)
