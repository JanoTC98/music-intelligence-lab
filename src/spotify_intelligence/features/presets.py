from __future__ import annotations

from pathlib import Path
from typing import Any

from spotify_intelligence.config import load_yaml_config
from spotify_intelligence.recommenders.errors import InvalidPreferenceProfileError

BASIC_FEATURES: tuple[str, ...] = (
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo",
)

# Ranges applied to preset values during validation (AGENTS.md sección 15.3).
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "energy": (0.0, 1.0),
    "danceability": (0.0, 1.0),
    "valence": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "tempo": (0.0, 300.0),
}

DEFAULT_WEIGHTS: dict[str, int] = {
    "energy": 2,
    "danceability": 2,
    "valence": 1,
    "acousticness": 2,
    "instrumentalness": 1,
    "tempo": 2,
}


def load_presets(
    path: str | Path = "configs/presets.yaml",
) -> dict[str, dict[str, Any]]:
    """Load and validate the preference presets from ``presets.yaml``.

    Returns a mapping ``{preset_key: {"label", "values", "weights"}}`` where
    ``values`` and ``weights`` always cover every ``BASIC_FEATURES``.
    """
    config = load_yaml_config(path)
    weight_scale = config.get("weight_scale", {})
    weight_min = int(weight_scale.get("min", 0))
    weight_max = int(weight_scale.get("max", 3))
    if weight_min != 0 or weight_max != 3:
        raise InvalidPreferenceProfileError(
            f"Unsupported weight scale {weight_min}..{weight_max}; expected 0..3"
        )

    presets = config.get("presets", {})
    if not presets:
        raise InvalidPreferenceProfileError("No presets defined in configuration")

    result: dict[str, dict[str, Any]] = {}
    for key, preset in presets.items():
        result[key] = _validate_preset(key, preset, weight_max)
    return result


def _validate_preset(
    key: str,
    preset: dict[str, Any],
    weight_max: int,
) -> dict[str, Any]:
    label = str(preset.get("label", key))
    raw_values = preset.get("values", {})
    raw_weights = preset.get("weights", {})

    values = {feature: float(raw_values[feature]) for feature in BASIC_FEATURES}
    for feature, (low, high) in VALUE_RANGES.items():
        if not low <= values[feature] <= high:
            raise InvalidPreferenceProfileError(
                f"Preset {key!r}: value for {feature!r} out of range [{low}, {high}]"
            )

    weights = {
        feature: int(raw_weights.get(feature, DEFAULT_WEIGHTS[feature]))
        for feature in BASIC_FEATURES
    }
    for feature, weight in weights.items():
        if not 0 <= weight <= weight_max:
            raise InvalidPreferenceProfileError(
                f"Preset {key!r}: weight for {feature!r} out of range [0, {weight_max}]"
            )

    return {"label": label, "values": values, "weights": weights}


def preset_names(path: str | Path = "configs/presets.yaml") -> list[str]:
    """Return the configured preset keys in stable order."""
    return list(load_presets(path).keys())


def get_preset(
    key: str,
    path: str | Path = "configs/presets.yaml",
) -> dict[str, Any]:
    """Return a single validated preset by key."""
    presets = load_presets(path)
    try:
        return presets[key]
    except KeyError:
        raise InvalidPreferenceProfileError(f"Unknown preset: {key!r}") from None
