"""Regression test for preset-driven widget reset (AGENTS.md §15/§18.5).

Switching presets must re-initialize the value sliders and weight selectboxes to
the newly selected preset. Otherwise Streamlit keeps the previous widget state
and the profile silently recommends with the wrong values. Requires the
versioned processed data and preference artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"

PRESET_ENERGY = {
    "entrenamiento_intenso": 0.90,
    "melancolico": 0.30,
    "relajacion": 0.25,
}


@pytest.fixture(scope="module")
def preferences_page() -> AppTest:
    """Return the preference page after its first run (default preset)."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    at.switch_page("app/pages/recommend_by_preferences.py")
    at.run()
    return at


def _energy_slider(at: AppTest) -> float:
    return float(at.slider(key="pref_value_energy").value)


def _valence_weight(at: AppTest) -> int:
    return int(at.selectbox(key="pref_weight_valence").value)


@pytest.mark.integration
def test_preset_switching_resets_value_and_weight_widgets(preferences_page: AppTest) -> None:
    """Selecting a preset loads its values and weights into the widgets."""

    assert _energy_slider(preferences_page) == pytest.approx(PRESET_ENERGY["entrenamiento_intenso"])

    preferences_page.selectbox(key="pref_preset").select("melancolico").run()
    assert _energy_slider(preferences_page) == pytest.approx(PRESET_ENERGY["melancolico"])
    assert _valence_weight(preferences_page) == 3

    preferences_page.selectbox(key="pref_preset").select("relajacion").run()
    assert _energy_slider(preferences_page) == pytest.approx(PRESET_ENERGY["relajacion"])
    assert _valence_weight(preferences_page) == 1
