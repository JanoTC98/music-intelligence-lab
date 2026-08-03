"""Home page: project overview and module status."""

from __future__ import annotations

import streamlit as st

from app.components import resources
from app.components.cards import metric_row
from app.components.messages import info_note


def render() -> None:
    config = resources.load_app_config()["app"]
    title = str(config.get("title", "Spotify Music Intelligence"))
    st.title(title)
    st.caption(
        "Recomendación musical basada en contenido y laboratorio de clasificación de géneros."
    )

    tracks = resources.load_tracks()
    recordings = resources.load_recordings()
    genres = resources.load_genre_catalog()

    metric_row(
        [
            ("Canciónes únicas", f"{len(tracks):,}"),
            ("Grabaciones (recording_group)", f"{len(recordings):,}"),
            ("Géneros", len(genres)),
        ]
    )

    st.markdown("### Módulos")
    st.markdown(
        """
- **Recomendar por canción** — grabaciones acústicamente cercanas a una canción.
- **Recomendar por preferencias** — presets editables y perfil manual ponderado.
- **Laboratorio multietiqueta** — Top-5 de géneros con umbral calibrado.
- **Laboratorio de género dominante** — género acústico dominante estimado.
- **Auditoría y catálogo** — calidad de datos y catálogo procesado.
- **Metodología y limitaciones** — decisiones y restricciones del proyecto.
"""
    )

    info_note(
        "Las similitudes y puntuaciones son medidas de distancia o de modelo; "
        "no representan probabilidades de gusto ni afiliación oficial con Spotify."
    )


if st.runtime.exists():
    render()
