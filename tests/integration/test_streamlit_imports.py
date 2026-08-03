"""Streamlit import smoke tests (AGENTS.md §25.7).

The router and every page are importable without a running Streamlit runtime,
perform no widget calls and never trigger training at import time.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import streamlit as st

PAGE_MODULES = [
    "app.pages.home",
    "app.pages.data_audit",
    "app.pages.recommend_by_track",
    "app.pages.recommend_by_preferences",
    "app.pages.multilabel_genre_lab",
    "app.pages.dominant_genre_lab",
    "app.pages.methodology",
]

COMPONENT_MODULES = [
    "app.components.resources",
    "app.components.cards",
    "app.components.charts",
    "app.components.filters",
    "app.components.messages",
    "app.components.tables",
]


@pytest.mark.parametrize("module_name", PAGE_MODULES)
def test_page_modules_importable(module_name: str) -> None:
    """Every page imports cleanly outside a Streamlit runtime."""
    module = importlib.import_module(module_name)
    assert callable(module.render)


@pytest.mark.parametrize("module_name", COMPONENT_MODULES)
def test_component_modules_importable(module_name: str) -> None:
    """Every shared component imports cleanly."""
    importlib.import_module(module_name)


def test_router_importable() -> None:
    """The router module is importable and exposes its guarded entrypoint."""
    module = importlib.import_module("streamlit_app")
    assert callable(module.main)


def test_no_runtime_in_pytest() -> None:
    """Pytest runs outside a Streamlit runtime, so guards are exercised."""
    assert not st.runtime.exists()


def test_importing_pages_does_not_render() -> None:
    """Importing pages must not execute their render() body.

    Because no Streamlit runtime exists, the ``if st.runtime.exists()`` guard
    prevents widget calls at import time. We assert the guard is present so the
    page files cannot silently execute under pytest.
    """
    for module_name in PAGE_MODULES:
        source = importlib.import_module(module_name).__file__
        assert source is not None
        text = Path(source).read_text(encoding="utf-8")
        assert "st.runtime.exists()" in text


def test_resources_are_cached() -> None:
    """Resource loaders expose the Streamlit cache wrappers (§18.2)."""
    resources = importlib.import_module("app.components.resources")
    for loader_name in (
        "load_tracks",
        "load_recordings",
        "load_genre_catalog",
        "load_track_recommender",
        "load_preference_recommender",
        "load_multilabel_serving",
        "load_multiclass_serving",
    ):
        loader = getattr(resources, loader_name)
        assert callable(loader)
        assert hasattr(loader, "clear")
