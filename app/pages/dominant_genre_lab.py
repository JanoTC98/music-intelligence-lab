"""Dominant genre lab page (AGENTS.md §17)."""

from __future__ import annotations

import streamlit as st

from app.components import charts, filters, messages, resources
from app.components.cards import seed_track_card
from spotify_intelligence.classification.serving import (
    load_validation_metrics,
    predict_multiclass_recording,
    select_recording_row,
)
from spotify_intelligence.data.contracts import DataContractError
from spotify_intelligence.recommenders.catalog import genres_for_recording
from spotify_intelligence.recommenders.errors import ArtifactNotFoundError


def render() -> None:
    st.title("Laboratorio de género dominante")
    st.caption(
        "Estima un género acústico dominante para una grabación usando el "
        "clasificador multiclase C1 (entrenado solo con grabaciones monoetiqueta)."
    )

    tracks = resources.load_tracks()
    recordings = resources.load_recordings()
    bridge = resources.load_genre_bridge()

    seed = filters.render_track_search(tracks, prefix="mc")
    if seed is None:
        st.caption("Selecciona una canción del catálogo.")
        return

    seed_track_card(seed)
    group = str(seed["recording_group_id"])
    recording_row = select_recording_row(recordings, group)

    try:
        serving = resources.load_multiclass_serving()
    except (ArtifactNotFoundError, DataContractError, FileNotFoundError, OSError):
        messages.missing_artifact("Clasificador de género dominante")
        return

    st.caption(f"Modelo {serving.model_id} · experimento {serving.experiment}")
    messages.render_validation_metrics(
        load_validation_metrics("multiclass", "C1"),
        kind="multiclass",
    )

    true_genres = genres_for_recording(bridge, group)
    if true_genres:
        st.markdown(f"**Etiquetas originales:** {', '.join(true_genres)}")
    else:
        st.markdown("**Etiquetas originales:** ninguna registrada.")

    if st.button("Estimar género dominante", type="primary", key="mc_run"):
        prediction = predict_multiclass_recording(serving, recording_row)
        items = [(item["genre"], item["score"]) for item in prediction["top_k"]]

        st.markdown("### Top-5 de géneros dominantes estimados")
        st.plotly_chart(
            charts.score_bars(items, title="Puntuaciones del modelo (no calibradas)"),
            width="stretch",
        )

        for item in prediction["top_k"]:
            st.markdown(f"- **{item['genre']}** · {item['score']:.4f}")

        if true_genres:
            top_names = [item["genre"] for item in prediction["top_k"]]
            hit_at_1 = top_names[0] in true_genres
            hit_at_3 = bool(set(top_names[:3]) & set(true_genres))
            st.markdown(
                f"- Hit@1 sobre etiquetas originales: **{hit_at_1}**  \n"
                f"- Hit@3 sobre etiquetas originales: **{hit_at_3}**"
            )

        messages.info_note(
            "El género estimado no reemplaza las etiquetas originales de una "
            "canción multigénero (§17.5). Las puntuaciones no están calibradas."
        )
        messages.warn_note(
            "Modelo experimental: con solo 18 características acústicas la "
            "separación entre géneros es limitada (§30), por lo que el género "
            "dominante estimado puede no coincidir con el género real de la canción."
        )


if st.runtime.exists():
    render()
