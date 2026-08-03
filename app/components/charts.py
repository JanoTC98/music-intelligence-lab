"""Plotly chart builders for the Streamlit app."""

from __future__ import annotations

import plotly.graph_objects as go


def _base_layout(title: str, x_title: str, y_title: str) -> dict:
    return {
        "title": title,
        "xaxis": {"title": x_title},
        "yaxis": {"title": y_title},
        "height": 360,
        "margin": {"l": 60, "r": 20, "t": 50, "b": 60},
        "showlegend": False,
    }


def score_bars(
    items: list[tuple[str, float]],
    *,
    title: str = "Puntuaciones",
    x_title: str = "Puntuación",
    y_title: str = "Género",
) -> go.Figure:
    """Return a horizontal bar chart of ``(label, score)`` pairs."""
    labels = [str(label) for label, _ in items]
    scores = [float(score) for _, score in items]
    figure = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                marker_color="#4C78A8",
            )
        ]
    )
    figure.update_layout(**_base_layout(title, x_title, y_title))
    figure.update_yaxes(autorange="reversed")
    return figure


def audit_counts(
    categories: list[str],
    values: list[int],
    *,
    title: str = "Recuentos",
    x_title: str = "Categoría",
    y_title: str = "Recuento",
) -> go.Figure:
    """Return a vertical bar chart of categorical counts."""
    figure = go.Figure(data=[go.Bar(x=categories, y=values, marker_color="#4C78A8")])
    figure.update_layout(**_base_layout(title, x_title, y_title))
    return figure
