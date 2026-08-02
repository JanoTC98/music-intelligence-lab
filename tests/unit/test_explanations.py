import pytest

from spotify_intelligence.recommenders.explanations import (
    explain_feature_differences,
    format_bpm_difference,
)


def test_explain_feature_differences_returns_rows():
    seed = {"energy": 0.8, "tempo": 120.0}
    candidate = {"energy": 0.7, "tempo": 116.0}
    rows = explain_feature_differences(seed, candidate)

    assert [r["feature"] for r in rows] == ["energy", "tempo"]
    assert rows[0]["difference"] == pytest.approx(-0.1)
    assert rows[0]["absolute_difference"] == pytest.approx(0.1)
    assert rows[1]["difference"] == pytest.approx(-4.0)


def test_explain_feature_differences_std_units():
    seed = {"energy": 0.8}
    candidate = {"energy": 0.7}
    rows = explain_feature_differences(seed, candidate, std_scale={"energy": 0.2})
    assert rows[0]["difference_std"] == pytest.approx(-0.5)


def test_explain_feature_differences_missing_feature_skipped():
    seed = {"energy": 0.8, "tempo": 120.0}
    candidate = {"energy": 0.7}
    rows = explain_feature_differences(seed, candidate)
    assert [r["feature"] for r in rows] == ["energy"]


def test_explain_feature_differences_labels():
    seed = {"energy": 0.8}
    candidate = {"energy": 0.7}
    rows = explain_feature_differences(seed, candidate, feature_labels={"energy": "Energía"})
    assert rows[0]["label"] == "Energía"


def test_format_bpm_difference():
    assert format_bpm_difference(0.5, std_bpm=9.6) == "4.8 BPM"


def test_format_bpm_difference_negative():
    assert format_bpm_difference(-0.4, std_bpm=10.0) == "-4.0 BPM"


def test_format_bpm_difference_none():
    assert format_bpm_difference(None, std_bpm=10.0) == ""
