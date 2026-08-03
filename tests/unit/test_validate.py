import pandas as pd
import pytest

from spotify_intelligence.data.contracts import (
    DataContractError,
    get_required_columns,
)
from spotify_intelligence.data.validate import (
    check_required_columns,
    detect_incomplete_audio,
    report_column_extremes,
    validate_column_ranges,
)


def test_check_required_columns_passes():
    df = pd.DataFrame({col: [] for col in get_required_columns()})
    result = check_required_columns(df)
    assert result == get_required_columns()


def test_check_required_columns_fails():
    df = pd.DataFrame({"track_id": [], "track_genre": []})
    with pytest.raises(DataContractError) as exc:
        check_required_columns(df)
    assert "Missing required columns" in str(exc.value)


def test_detect_incomplete_audio():
    df = pd.DataFrame(
        {
            "tempo": [0, 120, 0],
            "danceability": [0, 0.5, 0],
            "speechiness": [0, 0.1, 0],
            "valence": [0, 0.3, 0],
            "time_signature": [0, 4, 0],
        }
    )
    mask = detect_incomplete_audio(df)
    assert mask.tolist() == [True, False, True]


def test_detect_incomplete_audio_no_match():
    df = pd.DataFrame(
        {
            "tempo": [120],
            "danceability": [0.5],
            "speechiness": [0.1],
            "valence": [0.3],
            "time_signature": [4],
        }
    )
    mask = detect_incomplete_audio(df)
    assert mask.tolist() == [False]


def test_validate_column_ranges_popularity():
    df = pd.DataFrame({"popularity": [0, 50, 100, -1, 150]})
    violations = validate_column_ranges(df)
    assert "popularity" in violations
    violators = violations["popularity"]
    violator_values = [v for _, v in violators]
    assert -1 in violator_values
    assert 150 in violator_values


def test_validate_column_ranges_all_ok():
    df = pd.DataFrame({"popularity": [0, 50, 100]})
    violations = validate_column_ranges(df)
    assert "popularity" not in violations or len(violations["popularity"]) == 0


def test_validate_column_ranges_key():
    df = pd.DataFrame({"key": [0, 11, -1, 12, -2]})
    violations = validate_column_ranges(df)
    assert [v for _, v in violations["key"]] == [12, -2]


def test_validate_column_ranges_mode():
    df = pd.DataFrame({"mode": [0, 1, 0, 2]})
    violations = validate_column_ranges(df)
    assert [v for _, v in violations["mode"]] == [2]


def test_validate_column_ranges_tempo():
    df = pd.DataFrame({"tempo": [0, 120, -5]})
    violations = validate_column_ranges(df)
    assert [v for _, v in violations["tempo"]] == [-5]


def test_validate_column_ranges_time_signature():
    df = pd.DataFrame({"time_signature": [3, 4, 5, 6]})
    violations = validate_column_ranges(df)
    assert [v for _, v in violations["time_signature"]] == [6]


def test_report_column_extremes_loudness():
    df = pd.DataFrame({"loudness": [-10.5, -3.2, -20.0, None]})
    extremes = report_column_extremes(df)
    assert extremes["loudness"] == {"min": -20.0, "max": -3.2}


def test_report_column_extremes_empty():
    df = pd.DataFrame({"loudness": []})
    assert report_column_extremes(df) == {}
