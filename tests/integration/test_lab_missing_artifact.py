"""Regression test: a missing classifier artifact renders sección 18.4, no traceback.

The multilabel lab page offers the experimental B variant (imputation). If the
bundle is absent on disk (e.g. a stale or partial deploy), the page must show
the standard missing-artifact message instead of a raw traceback. The bundle
directory is renamed for the duration of the test and restored in teardown.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from streamlit.testing.v1 import AppTest

from app.components.messages import MISSING_ARTIFACT_HINT

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"
CLASSIFIER_DIR = Path("models/classifier") / "multilabel"
M1_B_SUFFIX = "_M1_B"


def _m1_b_dir() -> Path | None:
    if not CLASSIFIER_DIR.exists():
        return None
    for path in CLASSIFIER_DIR.iterdir():
        if path.is_dir() and path.name.endswith(M1_B_SUFFIX):
            return path
    return None


@pytest.fixture()
def hidden_m1_bundle() -> None:
    """Temporarily rename the M1_B bundle so it looks missing on disk."""
    bundle = _m1_b_dir()
    if bundle is None:
        pytest.skip("M1_B bundle not present locally")
    hidden = bundle.with_name(bundle.name + ".bak")
    bundle.rename(hidden)
    try:
        yield
    finally:
        hidden.rename(bundle)


@pytest.mark.integration
def test_multilabel_variant_b_shows_missing_artifact_message(
    hidden_m1_bundle: None,
) -> None:
    """Selecting variant B with a missing bundle shows the sección 18.4 message."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    at.switch_page("app/pages/multilabel_genre_lab.py")
    at.run()

    at.text_input(key="ml_query").set_value("Stressed Out").run()
    selected = at.selectbox(key="ml_selected")
    if not selected.options:
        pytest.skip("No seed track available for the search")
    selected.select(str(selected.options[0])).run()

    at.selectbox(key="ml_variant").select("B — imputación + indicador").run()

    assert not at.exception
    assert any(MISSING_ARTIFACT_HINT in element.value for element in at.error)
