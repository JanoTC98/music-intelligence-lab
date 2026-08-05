"""Track recommendation page."""

from __future__ import annotations

import streamlit as st

from app.components import filters, messages, resources, tables
from app.components.cards import seed_track_card
from spotify_intelligence.recommenders.errors import ArtifactNotFoundError

FEATURE_LABELS = {
    "danceability": "Bailabilidad",
    "energy": "Energía",
    "loudness": "Volumen",
    "speechiness": "Speechiness",
    "acousticness": "Acústica",
    "instrumentalness": "Instrumental",
    "liveness": "Liveness",
    "valence": "Valence",
    "tempo": "Tempo (BPM)",
}


def _format_explanation(differences: list[dict]) -> str:
    lines: list[str] = []
    for row in differences:
        label = FEATURE_LABELS.get(str(row.get("feature")), str(row.get("feature")))
        absolute = float(row.get("absolute_difference", 0.0))
        lines.append(f"{label}: diferencia absoluta {absolute:.3f}")
    return "\n".join(lines)


def render() -> None:
    st.title("Recomendar por canción")
    st.caption(
        "Encuentra grabaciones acústicamente cercanas a una canción del catálogo "
        "(unidad: recording_group_id)."
    )

    tracks = resources.load_tracks()
    genres = resources.load_genre_catalog()
    recommender_config = resources.load_recommender_config()
    retrieval = recommender_config["track_recommender"]["retrieval"]
    default_top_n = int(retrieval.get("default_top_n", 10))
    min_top_n = int(retrieval.get("min_top_n", 5))
    max_top_n = int(retrieval.get("max_top_n", 20))

    seed = filters.render_track_search(tracks, prefix="track")
    if seed is None:
        st.caption("Escribe una canción o artista para empezar.")
        return

    seed_track_card(seed)
    seed_group = str(seed["recording_group_id"])

    if bool(seed.get("audio_analysis_incomplete", False)):
        messages.warn_note(
            "Esta grabación tiene análisis acústico incompleto y no puede usarse como semilla."
        )
        return

    try:
        recommender = resources.load_track_recommender()
    except ArtifactNotFoundError:
        messages.missing_artifact("Recomendador por canción")
        return

    top_n = st.slider(
        "Número de resultados",
        min_value=min_top_n,
        max_value=max_top_n,
        value=default_top_n,
    )
    recommendation_filters = filters.render_recommendation_filters(
        recommender_config, genres, prefix="track"
    )

    affinity_config = recommender_config["track_recommender"].get("genre_affinity", {})
    genre_affinity = st.toggle(
        "Priorizar canciones del mismo género de la semilla",
        value=bool(affinity_config.get("enabled_default", False)),
        help="Reordena los candidatos acústicamente más cercanos para que "
        "compartan género con la semilla (variante experimental).",
        key="track_genre_affinity",
    )

    if st.button("Recomendar", type="primary", key="track_run"):
        results = recommender.recommend(
            seed_group,
            top_n=top_n,
            filters=recommendation_filters,
            genre_affinity=genre_affinity,
        )
        if results.empty:
            messages.empty_state("No hay resultados con los filtros seleccionados.")
            return

        display = tables.result_display_frame(results)
        st.markdown(f"### {len(results)} recomendaciones")
        st.dataframe(display, width="stretch")
        tables.download_csv_button(display, "recomendaciones_por_cancion.csv")

        with st.expander("Explicaciones por característica"):
            for position, row in results.iterrows():
                differences = row.get("feature_differences") or []
                st.markdown(
                    f"**{position + 1}. {row['track_name']}** — {row['artists']}  \n"
                    f"Similitud coseno: {float(row['similarity']):.3f}"
                )
                if differences:
                    st.code(_format_explanation(differences), language="text")
                st.divider()

        messages.info_note(
            "La similitud coseno es una medida de distancia acústica, no una probabilidad de gusto."
        )


if st.runtime.exists():
    render()
