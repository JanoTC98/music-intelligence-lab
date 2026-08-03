"""Simple info cards for tracks and results."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.tables import format_duration


def seed_track_card(row: Any) -> None:
    """Render a compact header card for the selected seed recording."""
    track_name = str(row.get("track_name", "—"))
    artists = str(row.get("artists", "—"))
    album = str(row.get("album_name", "—"))
    duration = format_duration(row.get("duration_ms"))
    popularity = row.get("popularity_median")

    st.markdown(
        f"**{track_name}** — {artists}  \n"
        f"Álbum: {album} · Duración: {duration} · "
        f"Popularidad: {popularity if popularity is not None else '—'}"
    )


def metric_row(cells: list[tuple[str, object]]) -> None:
    """Render a row of ``st.metric`` cells."""
    columns = st.columns(len(cells))
    for column, (label, value) in zip(columns, cells, strict=False):
        column.metric(label, str(value))
