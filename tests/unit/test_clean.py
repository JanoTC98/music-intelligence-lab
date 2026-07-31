import pandas as pd

from spotify_intelligence.data.clean import (
    add_anomaly_flags,
    build_genre_catalog,
    build_track_artists,
    build_track_catalog,
    build_track_genres,
    clean_records,
)

AUDIO_COLS = {
    "danceability": 0.5,
    "energy": 0.6,
    "key": 4,
    "loudness": -8.0,
    "mode": 1,
    "speechiness": 0.05,
    "acousticness": 0.3,
    "instrumentalness": 0.0,
    "liveness": 0.2,
    "valence": 0.4,
    "tempo": 120.0,
    "time_signature": 4,
}


def make_rows():
    return [
        {
            "Unnamed: 0": 0,
            "track_id": "t1",
            "track_name": "Song A",
            "artists": "Artist One",
            "album_name": "Album A",
            "popularity": 50,
            "duration_ms": 200000,
            "explicit": False,
            "track_genre": "pop",
            **AUDIO_COLS,
        },
        {
            "Unnamed: 0": 1,
            "track_id": "t2",
            "track_name": "Song B",
            "artists": "Artist Two",
            "album_name": "Album B",
            "popularity": 30,
            "duration_ms": 45000,
            "explicit": True,
            "track_genre": "rock",
            **AUDIO_COLS,
        },
        {
            "Unnamed: 0": 2,
            "track_id": "t3",
            "track_name": "Song C",
            "artists": "Artist Three",
            "album_name": "Album C",
            "popularity": 10,
            "duration_ms": 700000,
            "explicit": False,
            "track_genre": "jazz",
            **AUDIO_COLS,
        },
    ]


def make_invalid_row():
    return {
        "Unnamed: 0": 3,
        "track_id": "tbad",
        "track_name": None,
        "artists": "Artist Bad",
        "album_name": "Album Bad",
        "popularity": 10,
        "duration_ms": 700000,
        "explicit": False,
        "track_genre": "jazz",
        **AUDIO_COLS,
    }


def test_clean_records_quarantines_invalid_identity():
    df = pd.DataFrame(make_rows() + [make_invalid_row()])
    valid, invalid = clean_records(df)
    assert len(valid) == 3
    assert len(invalid) == 1
    assert invalid.iloc[0]["track_id"] == "tbad"


def test_clean_records_drops_index_column():
    df = pd.DataFrame(make_rows())
    valid, _ = clean_records(df)
    assert "Unnamed: 0" not in valid.columns


def test_add_anomaly_flags_marks_flags():
    df = pd.DataFrame(make_rows())
    valid, _ = clean_records(df)
    result = add_anomaly_flags(valid)
    by_id = result.set_index("track_id")
    assert not by_id.loc["t1", "is_short_track"]
    assert not by_id.loc["t1", "is_long_track"]
    assert by_id.loc["t2", "is_short_track"]
    assert by_id.loc["t3", "is_long_track"]


def test_add_anomaly_flags_incomplete_audio():
    row = {
        "track_id": "t4",
        "track_name": "Song D",
        "artists": "Artist Four",
        "album_name": "Album D",
        "popularity": 0,
        "duration_ms": 200000,
        "explicit": False,
        "track_genre": "pop",
        "danceability": 0.0,
        "energy": 0.0,
        "key": 0,
        "loudness": -30.0,
        "mode": 0,
        "speechiness": 0.0,
        "acousticness": 0.0,
        "instrumentalness": 0.0,
        "liveness": 0.0,
        "valence": 0.0,
        "tempo": 0.0,
        "time_signature": 0,
    }
    result = add_anomaly_flags(pd.DataFrame([row]))
    assert result.iloc[0]["audio_analysis_incomplete"]


def test_build_track_catalog_one_row_per_track_id():
    rows = make_rows()
    rows.append({**rows[0], "Unnamed: 0": 3, "popularity": 80, "track_genre": "dance"})
    df = pd.DataFrame(rows)
    valid, _ = clean_records(df)
    valid = add_anomaly_flags(valid)
    catalog = build_track_catalog(valid)
    assert len(catalog) == 3
    assert catalog["track_id"].is_unique
    t1 = catalog[catalog["track_id"] == "t1"].iloc[0]
    assert t1["popularity_min"] == 50
    assert t1["popularity_max"] == 80
    assert t1["popularity_observations"] == 2
    assert round(t1["duration_min"], 4) == round(200000 / 60000.0, 4)


def test_build_track_genres_deduplicates():
    rows = make_rows()
    rows.append({**rows[0], "Unnamed: 0": 3, "popularity": 80, "track_genre": "dance"})
    df = pd.DataFrame(rows)
    valid, _ = clean_records(df)
    genres = build_track_genres(valid)
    assert len(genres) == 4
    assert not genres.duplicated(subset=["track_id", "track_genre"]).any()


def test_build_genre_catalog():
    df = pd.DataFrame(make_rows())
    valid, _ = clean_records(df)
    catalog = build_genre_catalog(valid)
    assert len(catalog) == 3
    assert set(catalog["track_genre"]) == {"pop", "rock", "jazz"}


def test_build_track_artists_splits_on_semicolon():
    rows = make_rows()
    rows.append(
        {
            **rows[0],
            "Unnamed: 0": 3,
            "track_id": "t5",
            "artists": "Artist One;Artist Two",
            "popularity": 20,
        }
    )
    df = pd.DataFrame(rows)
    valid, _ = clean_records(df)
    artists = build_track_artists(valid)
    t5 = artists[artists["track_id"] == "t5"]
    assert len(t5) == 2
    assert set(t5["artist"]) == {"Artist One", "Artist Two"}
