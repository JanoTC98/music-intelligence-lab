"""Widget builders for recommendation filters and preference profiles."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from spotify_intelligence.features.presets import (
    BASIC_FEATURES,
    DEFAULT_WEIGHTS,
    VALUE_RANGES,
)
from spotify_intelligence.recommenders.catalog import search_catalog
from spotify_intelligence.recommenders.preference_based import PreferenceProfile
from spotify_intelligence.recommenders.track_based import RecommendationFilters

_EXPLICIT_LABELS = {"all": "Todos", "explicit": "Explícitas", "non_explicit": "No explícitas"}

FEATURE_LABELS = {
    "energy": "Energía",
    "danceability": "Bailabilidad",
    "valence": "Valence (positividad)",
    "acousticness": "Acústica",
    "instrumentalness": "Instrumental",
    "tempo": "Tempo (BPM)",
}


def _explicit_format(value: str) -> str:
    return _EXPLICIT_LABELS.get(value, value)


def render_recommendation_filters(
    config: dict[str, Any],
    genres: list[str],
    *,
    prefix: str = "filters",
    include_different_artist: bool = True,
) -> RecommendationFilters:
    """Render optional recommendation filters (AGENTS.md §14.8).

    Every filter is inactive by default; the widget state always matches the
    approved configuration values from ``recommender_features.yaml``. The
    ``different_artist`` filter is only meaningful for track-based seeds and can
    be hidden on other pages.
    """
    filters_config = config["track_recommender"]["filters"]
    duration_min = int(filters_config.get("duration_suggested_min_seconds", 60))
    duration_max = int(filters_config.get("duration_suggested_max_seconds", 600))

    with st.expander("Filtros opcionales", expanded=False):
        explicit = st.selectbox(
            "Contenido explícito",
            options=["all", "explicit", "non_explicit"],
            index=0,
            format_func=_explicit_format,
            key=f"{prefix}_explicit",
        )
        selected_genres = st.multiselect(
            "Géneros",
            options=genres,
            key=f"{prefix}_genres",
        )
        enable_duration = st.checkbox(
            "Limitar duración",
            value=False,
            key=f"{prefix}_duration_enabled",
        )
        duration_range = st.slider(
            "Duración sugerida (segundos)",
            min_value=30,
            max_value=900,
            value=(duration_min, duration_max),
            key=f"{prefix}_duration_range",
        )
        different_artist = st.checkbox(
            "Excluir artistas de la semilla",
            value=False,
            key=f"{prefix}_different_artist",
            disabled=not include_different_artist,
        )
        enable_popularity = st.checkbox(
            "Popularidad mínima",
            value=False,
            key=f"{prefix}_popularity_enabled",
        )
        popularity_min = st.slider(
            "Popularidad mínima",
            min_value=0,
            max_value=100,
            value=int(filters_config.get("popularity_min_suggested", 50)),
            key=f"{prefix}_popularity_min",
        )

    return RecommendationFilters(
        explicit=str(explicit),
        genres=list(selected_genres) if selected_genres else None,
        duration_enabled=bool(enable_duration),
        duration_min_seconds=int(duration_range[0]),
        duration_max_seconds=int(duration_range[1]),
        different_artist=bool(different_artist),
        popularity_min=int(popularity_min) if enable_popularity else None,
    )


def render_preference_profile(
    presets: dict[str, dict[str, Any]],
    *,
    prefix: str = "profile",
) -> PreferenceProfile:
    """Render preset + editable sliders and weights (§15.2/§15.5).

    The returned profile always validates: at least one weight is non-zero and
    every value stays inside its approved range.
    """
    preset_keys = list(presets.keys())
    preset_labels = {key: presets[key]["label"] for key in preset_keys}
    preset_key = st.selectbox(
        "Preset",
        options=preset_keys,
        index=0,
        format_func=lambda key: preset_labels.get(key, key),
        key=f"{prefix}_preset",
    )
    selected = presets[preset_key]

    last_preset_key = f"{prefix}_last_preset"
    if st.session_state.get(last_preset_key) != preset_key:
        for feature in BASIC_FEATURES:
            st.session_state.pop(f"{prefix}_value_{feature}", None)
            st.session_state.pop(f"{prefix}_weight_{feature}", None)
        st.session_state[last_preset_key] = preset_key

    with st.expander("Editar valores", expanded=True):
        values: dict[str, float] = {}
        for feature in BASIC_FEATURES:
            low, high = VALUE_RANGES[feature]
            step = 1.0 if feature == "tempo" else 0.01
            values[feature] = st.slider(
                FEATURE_LABELS.get(feature, feature),
                min_value=float(low),
                max_value=float(high),
                value=float(selected["values"][feature]),
                step=float(step),
                key=f"{prefix}_value_{feature}",
            )

    with st.expander("Pesos (0 = ignorar, 3 = alta importancia)", expanded=True):
        weights: dict[str, int] = {}
        for feature in BASIC_FEATURES:
            weights[feature] = st.selectbox(
                f"Peso · {FEATURE_LABELS.get(feature, feature)}",
                options=[0, 1, 2, 3],
                index=int(selected["weights"].get(feature, DEFAULT_WEIGHTS[feature])),
                key=f"{prefix}_weight_{feature}",
            )

    return PreferenceProfile.from_manual(values, weights, label=preset_labels.get(preset_key))


def render_track_search(
    tracks: pd.DataFrame,
    *,
    prefix: str = "search",
    label: str = "Buscar canción o artista",
) -> pd.Series | None:
    """Render a disambiguated track search and return the selected row.

    Returns ``None`` while no query has produced a confirmed selection. The
    caller decides what to render next (AGENTS.md §18.5 "búsqueda
    desambiguada por canción y artista").
    """
    query = st.text_input(label, key=f"{prefix}_query")
    if not query.strip():
        return None

    results = search_catalog(tracks, query)
    if results.empty:
        st.info("Sin coincidencias. Prueba con otro título o artista.")
        return None

    st.caption(
        f"Se encontraron {len(results)} coincidencias — usa el desplegable para verlas todas."
    )

    options = results["track_id"].tolist()
    mapping = results.set_index("track_id")
    labels = {
        track_id: " — ".join(
            part
            for part in (
                str(row["track_name"]).strip(),
                str(row["artists"]).strip(),
                str(row["album_name"]).strip(),
            )
            if part
        )
        for track_id, row in mapping.iterrows()
    }
    selected = st.selectbox(
        "Selecciona la canción exacta",
        options=options,
        index=0,
        format_func=lambda track_id: labels.get(track_id, track_id),
        key=f"{prefix}_selected",
    )
    return mapping.loc[selected]
