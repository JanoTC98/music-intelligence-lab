import pytest

from spotify_intelligence.features.presets import (
    BASIC_FEATURES,
    load_presets,
    preset_names,
)
from spotify_intelligence.recommenders.errors import InvalidPreferenceProfileError


def test_all_presets_loaded_from_config():
    assert set(preset_names()) == {
        "entrenamiento_intenso",
        "fiesta",
        "concentracion_instrumental",
        "relajacion",
        "alegre_y_bailable",
        "melancolico",
        "acustico",
    }


def test_presets_cover_all_basic_features():
    presets = load_presets()
    for preset in presets.values():
        assert set(preset["values"].keys()) == set(BASIC_FEATURES)
        assert set(preset["weights"].keys()) == set(BASIC_FEATURES)


def test_preset_values_within_ranges():
    presets = load_presets()
    for preset in presets.values():
        for feature in ["energy", "danceability", "valence", "acousticness", "instrumentalness"]:
            assert 0.0 <= preset["values"][feature] <= 1.0
        assert 0.0 <= preset["values"]["tempo"] <= 300.0


def test_preset_weights_within_scale():
    presets = load_presets()
    for preset in presets.values():
        for weight in preset["weights"].values():
            assert 0 <= weight <= 3


def test_unknown_preset_raises():
    from spotify_intelligence.features.presets import get_preset

    with pytest.raises(InvalidPreferenceProfileError):
        get_preset("no_existe")
