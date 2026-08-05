"""Multilabel genre lab page."""

from __future__ import annotations

import streamlit as st

from app.components import charts, filters, messages, resources
from app.components.cards import seed_track_card
from spotify_intelligence.classification.serving import (
    load_validation_metrics,
    predict_multilabel_recording,
    select_recording_row,
)
from spotify_intelligence.data.contracts import DataContractError
from spotify_intelligence.recommenders.catalog import genres_for_recording
from spotify_intelligence.recommenders.errors import ArtifactNotFoundError

VARIANTS = {
    "A — excluye audio incompleto (baseline)": "M1_A",
    "B — imputación + indicador": "M1_B",
}


def render() -> None:
    st.title("Laboratorio multietiqueta")
    st.caption(
        "Predice un conjunto compatible de los 114 géneros a partir de "
        "características acústicas (Top-5 con umbral calibrado)."
    )

    tracks = resources.load_tracks()
    recordings = resources.load_recordings()
    bridge = resources.load_genre_bridge()

    seed = filters.render_track_search(tracks, prefix="ml")
    if seed is None:
        st.caption("Selecciona una canción del catálogo.")
        return

    seed_track_card(seed)
    group = str(seed["recording_group_id"])
    recording_row = select_recording_row(recordings, group)

    variant_label = st.selectbox(
        "Variante de datos",
        options=list(VARIANTS.keys()),
        index=0,
        key="ml_variant",
    )
    model_key = VARIANTS[variant_label]

    try:
        serving = resources.load_multilabel_serving(model_key)
    except (ArtifactNotFoundError, DataContractError, FileNotFoundError, OSError):
        messages.missing_artifact("Clasificador multietiqueta")
        return

    st.caption(
        f"Modelo {serving.model_id} · experimento {serving.experiment} · "
        f"umbral {serving.threshold:.2f}"
    )
    messages.render_validation_metrics(
        load_validation_metrics("multilabel", model_key),
        kind="multilabel",
    )

    true_genres = genres_for_recording(bridge, group)
    if true_genres:
        st.markdown(f"**Etiquetas originales:** {', '.join(true_genres)}")
    else:
        st.markdown("**Etiquetas originales:** ninguna registrada.")

    if bool(recording_row.get("audio_analysis_incomplete", False)):
        messages.warn_note(
            "Esta grabación tiene análisis acústico incompleto. En la "
            "variante A el modelo no vio este patrón durante el entrenamiento."
        )

    if st.button("Predecir géneros", type="primary", key="ml_run"):
        prediction = predict_multilabel_recording(serving, recording_row)
        items = [(item["genre"], item["score"]) for item in prediction["top_k"]]

        st.markdown("### Top-5 de géneros")
        st.plotly_chart(
            charts.score_bars(items, title="Puntuaciones del modelo (no calibradas)"),
            width="stretch",
        )

        for item in prediction["top_k"]:
            st.markdown(f"- **{item['genre']}** · {item['score']:.4f}")

        if prediction["below_threshold"]:
            messages.warn_note(
                f"Ninguna etiqueta superó el umbral {prediction['threshold']:.2f}; "
                "se muestra el top-1 con la advertencia correspondiente."
            )

        messages.info_note(
            "Las puntuaciones no están calibradas; no deben leerse como probabilidades."
        )
        messages.warn_note(
            "Modelo experimental: con solo 18 características acústicas la "
            "separación entre géneros es limitada, por lo que las "
            "predicciones pueden no coincidir con los géneros reales de la canción."
        )


if st.runtime.exists():
    render()
