"""Streamlit entrypoint / router (AGENTS.md sección 18.1).

The script only builds navigation when a Streamlit runtime is active, so
importing it from pytest is safe and performs no widget calls (sección 25.7).
"""

from __future__ import annotations

import streamlit as st

from spotify_intelligence.config import load_yaml_config

APP_CONFIG_PATH = "configs/app.yaml"


def main() -> None:
    app_config = load_yaml_config(APP_CONFIG_PATH)["app"]

    st.set_page_config(
        page_title=str(app_config.get("title", "Spotify Music Intelligence")),
        page_icon=str(app_config.get("page_icon", "🎵")),
        layout=str(app_config.get("layout", "wide")),
    )

    pages = [
        st.Page("app/pages/home.py", title="Inicio", icon="🏠", default=True),
        st.Page(
            "app/pages/data_audit.py",
            title="Auditoría y catálogo",
            icon="📊",
        ),
        st.Page(
            "app/pages/recommend_by_track.py",
            title="Recomendar por canción",
            icon="🎧",
        ),
        st.Page(
            "app/pages/recommend_by_preferences.py",
            title="Recomendar por preferencias",
            icon="🎛️",
        ),
        st.Page(
            "app/pages/multilabel_genre_lab.py",
            title="Laboratorio multietiqueta",
            icon="🏷️",
        ),
        st.Page(
            "app/pages/dominant_genre_lab.py",
            title="Laboratorio de género dominante",
            icon="🎼",
        ),
        st.Page(
            "app/pages/methodology.py",
            title="Metodología y limitaciones",
            icon="📚",
        ),
    ]

    navigation = st.navigation(pages)
    navigation.run()


if st.runtime.exists():
    main()
