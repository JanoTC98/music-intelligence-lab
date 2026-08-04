"""Cached resource loaders for the Streamlit app (AGENTS.md sección 18.2).

Data tables use ``st.cache_data``; model instances, scalers and indices use
``st.cache_resource``. The app never trains or fits anything at runtime
(sección 18.4); missing artifacts surface as exceptions that pages convert into
user-facing messages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spotify_intelligence.classification.serving import (
    load_multiclass_serving as _load_multiclass_serving,
)
from spotify_intelligence.classification.serving import (
    load_multilabel_serving as _load_multilabel_serving,
)
from spotify_intelligence.classification.serving import (
    load_recordings as _load_recordings,
)
from spotify_intelligence.config import load_yaml_config
from spotify_intelligence.features.presets import load_presets
from spotify_intelligence.recommenders.catalog import (
    genres_by_recording_group,
    read_catalog_tracks,
    read_genre_bridge,
)
from spotify_intelligence.recommenders.preference_based import PreferenceRecommender
from spotify_intelligence.recommenders.track_based import TrackRecommender

PROCESSED_DIR = Path("data/processed")
RECOMMENDER_DIR = Path("models/recommender/v1")
PREFERENCES_DIR = Path("models/preferences/v1")
CLASSIFIER_DIR = Path("models/classifier")
REPORT_DIR = Path("reports/data_quality")

APP_CONFIG_PATH = Path("configs/app.yaml")
PRESETS_PATH = Path("configs/presets.yaml")
RECOMMENDER_FEATURES_PATH = Path("configs/recommender_features.yaml")


@st.cache_data(show_spinner=False)
def load_app_config() -> dict:
    """Load ``configs/app.yaml`` (title, defaults, tracking flags)."""
    return load_yaml_config(APP_CONFIG_PATH)


@st.cache_data(show_spinner=False)
def load_recommender_config() -> dict:
    """Load ``configs/recommender_features.yaml`` (features, filters, retrieval)."""
    return load_yaml_config(RECOMMENDER_FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_presets_config() -> dict[str, dict]:
    """Load the validated preference presets from ``configs/presets.yaml``."""
    return load_presets(PRESETS_PATH)


@st.cache_data(show_spinner=False)
def load_tracks() -> pd.DataFrame:
    """Load the unique track catalog used by the search index."""
    return read_catalog_tracks(PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def load_recordings() -> pd.DataFrame:
    """Load the canonical recording catalog."""
    return _load_recordings(PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def load_genre_bridge() -> pd.DataFrame:
    """Load the recording-group to genre bridge table."""
    return read_genre_bridge(PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def load_genres_by_group() -> dict[str, list[str]]:
    """Return ``{recording_group_id: [genre, ...]}`` for display."""
    return genres_by_recording_group(load_genre_bridge())


@st.cache_data(show_spinner=False)
def load_genre_catalog() -> list[str]:
    """Return the 114 canonical genre names in genre_id order."""
    path = PROCESSED_DIR / "genre_catalog.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    return [str(value) for value in frame["track_genre"].tolist()]


@st.cache_data(show_spinner=False)
def load_data_quality_report() -> dict | None:
    """Load the reproducible audit report if present."""
    path = REPORT_DIR / "data_quality_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_track_recommender() -> TrackRecommender:
    """Load the versioned track recommender (R1 baseline)."""
    return TrackRecommender(RECOMMENDER_DIR)


@st.cache_resource(show_spinner=False)
def load_preference_recommender() -> PreferenceRecommender:
    """Load the versioned preference recommender."""
    return PreferenceRecommender(PREFERENCES_DIR)


@st.cache_resource(show_spinner=False)
def load_multilabel_serving(model_key: str = "M1_A"):
    """Load a multilabel serving bundle (variant A or B)."""
    return _load_multilabel_serving(model_key, CLASSIFIER_DIR)


@st.cache_resource(show_spinner=False)
def load_multiclass_serving():
    """Load the final dominant-genre serving bundle (C1)."""
    return _load_multiclass_serving("C1", CLASSIFIER_DIR)
