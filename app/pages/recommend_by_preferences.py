"""Preference recommendation page."""

from __future__ import annotations

import streamlit as st

from app.components import filters, messages, resources, tables
from spotify_intelligence.recommenders.errors import (
    ArtifactNotFoundError,
    InvalidPreferenceProfileError,
)

OOD_MESSAGES = {
    "warning": "El perfil es poco frecuente en el catálogo (percentil 95).",
    "weak_match": "El perfil está muy lejos del catálogo (percentil 99); las coincidencias serán débiles.",
}


def render() -> None:
    st.title("Recomendar por preferencias")
    st.caption(
        "Encuentra canciones cercanas a un perfil acústico ponderado, "
        "partiendo de un preset editable o de un perfil manual."
    )

    genres = resources.load_genre_catalog()
    recommender_config = resources.load_recommender_config()
    retrieval = recommender_config["track_recommender"]["retrieval"]
    default_top_n = int(retrieval.get("default_top_n", 10))
    min_top_n = int(retrieval.get("min_top_n", 5))
    max_top_n = int(retrieval.get("max_top_n", 20))

    try:
        presets = resources.load_presets_config()
    except InvalidPreferenceProfileError as error:
        messages.error_note(f"Configuración de presets inválida: {error}")
        return

    profile = filters.render_preference_profile(presets, prefix="pref")

    try:
        recommender = resources.load_preference_recommender()
    except ArtifactNotFoundError:
        messages.missing_artifact("Recomendador por preferencias")
        return

    ood = recommender.out_of_distribution_status(profile)
    if ood["status"] == "warning":
        messages.warn_note(OOD_MESSAGES["warning"])
    elif ood["status"] == "weak_match":
        messages.warn_note(OOD_MESSAGES["weak_match"])

    recommendation_filters = filters.render_recommendation_filters(
        recommender_config,
        genres,
        prefix="pref",
        include_different_artist=False,
    )

    top_n = st.slider(
        "Número de resultados",
        min_value=min_top_n,
        max_value=max_top_n,
        value=default_top_n,
    )

    diversity_config = recommender_config["preference_recommender"]["diversity"]
    with st.expander("Diversidad (MMR)", expanded=False):
        diversity_enabled = st.checkbox(
            "Activar reordenamiento por diversidad",
            value=False,
            key="pref_diversity",
        )
        lambda_value = st.slider(
            "Lambda (relevancia vs. diversidad)",
            min_value=0.50,
            max_value=1.00,
            value=float(diversity_config.get("lambda_default", 0.85)),
            step=0.05,
            key="pref_lambda",
        )

    if st.button("Recomendar", type="primary", key="pref_run"):
        try:
            results = recommender.recommend(
                profile,
                top_n=top_n,
                filters=recommendation_filters,
                diversity_enabled=diversity_enabled,
                lambda_=lambda_value,
            )
        except InvalidPreferenceProfileError as error:
            messages.error_note(str(error))
            return

        if results.empty:
            messages.empty_state("No hay resultados con los filtros seleccionados.")
            return

        display = tables.result_display_frame(results)
        st.markdown(f"### {len(results)} recomendaciones")
        st.dataframe(display, width="stretch")
        tables.download_csv_button(display, "recomendaciones_por_preferencias.csv")

        messages.info_note(
            "La similitud se deriva de la distancia ponderada; no es una probabilidad de gusto."
        )


if st.runtime.exists():
    render()
