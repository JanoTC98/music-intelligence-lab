"""Data audit and catalog page."""

from __future__ import annotations

import streamlit as st

from app.components import charts, resources
from app.components.cards import metric_row
from app.components.messages import error_note


def render() -> None:
    st.title("Auditoría y catálogo")
    st.caption("Resumen reproducible de la auditoría del dataset y del catálogo procesado.")

    report = resources.load_data_quality_report()
    if report is None:
        error_note("No se encontró reports/data_quality/data_quality_report.json.")
        return

    st.subheader("Dataset original")
    metric_row(
        [
            ("Filas", f"{int(report.get('raw_rows', 0)):,}"),
            ("Columnas", int(report.get("raw_columns", 0))),
            ("Géneros", int(report.get("genres", {}).get("total_unique", 0))),
            ("Hash (SHA-256)", str(report.get("dataset_hash", ""))[:16] + "…"),
        ]
    )

    with st.expander("Calidad e integridad", expanded=True):
        quality_cols = st.columns(3)
        quality_cols[0].markdown(
            f"**Columnas requeridas:** {'OK' if report.get('required_columns_ok') else 'FALLO'}"
        )
        nulls = report.get("nulls", {})
        quality_cols[1].markdown(f"**Celdas nulas:** {int(nulls.get('total_cells', 0))}")
        duplicates = report.get("duplicates", {})
        quality_cols[2].markdown(
            f"**Duplicados de track_id:** {int(duplicates.get('track_id_duplicates', 0))}"
        )

    st.subheader("Anomalías detectadas")
    incomplete = report.get("incomplete_audio", {})
    duration = report.get("duration", {})
    popularity = report.get("popularity", {})
    multi_genre = report.get("multi_genre", {})

    anomaly_categories = ["Audio incompleto", "Cortas (<60s)", "Largas (>10min)", "Popularidad 0"]
    anomaly_values = [
        int(incomplete.get("count", 0)),
        int(duration.get("short_tracks_under_60s", 0)),
        int(duration.get("long_tracks_over_10min", 0)),
        int(popularity.get("zero_count", 0)),
    ]
    st.plotly_chart(
        charts.audit_counts(
            anomaly_categories,
            anomaly_values,
            title="Anomalías por categoría",
        ),
        width="stretch",
    )
    st.markdown(
        f"- Canciones multigénero: **{int(multi_genre.get('tracks_with_multiple_genres', 0)):,}** "
        f"(máx. {int(multi_genre.get('max_genres_per_track', 0))} géneros)."
    )

    st.subheader("Catálogo procesado")
    tracks = resources.load_tracks()
    recordings = resources.load_recordings()
    metric_row(
        [
            ("track_id únicos", f"{len(tracks):,}"),
            ("recording_group_id", f"{len(recordings):,}"),
        ]
    )
    st.markdown(
        "Los datos originales nunca se modifican; el catálogo es la fuente única "
        "para recomendadores, clasificadores y aplicación."
    )


if st.runtime.exists():
    render()
