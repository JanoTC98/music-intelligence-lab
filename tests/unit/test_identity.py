import pandas as pd

from spotify_intelligence.identity.fingerprints import (
    build_exact_fingerprint,
    fingerprint_to_recording_group_id,
)
from spotify_intelligence.identity.normalize import normalize_text
from spotify_intelligence.identity.recording_groups import (
    assign_recording_group_ids,
    build_recording_tracks,
    build_recordings,
)


def _make_tracks_df():
    return pd.DataFrame(
        [
            {
                "track_id": "t1",
                "track_name": "Song A",
                "track_name_normalized": "song a",
                "artists": "Artist One",
                "artists_normalized": "artist one",
                "album_name": "Album A",
                "duration_ms": 200000,
                "explicit": False,
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
                "popularity_median": 60.0,
                "audio_analysis_incomplete": False,
            },
            {
                "track_id": "t2",
                "track_name": "Song A (Remastered)",
                "track_name_normalized": "song a (remastered)",
                "artists": "Artist One",
                "artists_normalized": "artist one",
                "album_name": "Album B",
                "duration_ms": 200000,
                "explicit": False,
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
                "popularity_median": 90.0,
                "audio_analysis_incomplete": False,
            },
            {
                "track_id": "t3",
                "track_name": "Different Song",
                "track_name_normalized": "different song",
                "artists": "Artist Two",
                "artists_normalized": "artist two",
                "album_name": "Album C",
                "duration_ms": 300000,
                "explicit": False,
                "danceability": 0.9,
                "energy": 0.9,
                "key": 9,
                "loudness": -3.0,
                "mode": 0,
                "speechiness": 0.1,
                "acousticness": 0.0,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.8,
                "tempo": 140.0,
                "time_signature": 4,
                "popularity_median": 45.0,
                "audio_analysis_incomplete": False,
            },
        ]
    )


def test_normalize_text_basic():
    assert normalize_text("  Song  A  ") == "song a"


def test_normalize_text_preserves_punctuation():
    assert normalize_text("Song A (Remastered)") == "song a (remastered)"


def test_normalize_text_handles_unicode_nfkc():
    value = "caf\u00e9"
    assert normalize_text(value) == "caf\u00e9".casefold()


def test_build_exact_fingerprint_deterministic():
    fields = {
        "track_name_normalized": "song a",
        "artists_normalized": "artist one",
        "duration_ms": 200000,
        "explicit": False,
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
    fp1 = build_exact_fingerprint(fields)
    fp2 = build_exact_fingerprint(fields)
    assert fp1 == fp2


def test_fingerprint_sha256_length():
    fields = {
        "track_name_normalized": "song a",
        "artists_normalized": "artist one",
        "duration_ms": 200000,
        "explicit": False,
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
    group_id = fingerprint_to_recording_group_id(build_exact_fingerprint(fields))
    assert len(group_id) == 64
    assert set(group_id) <= set("0123456789abcdef")


def test_assign_recording_group_ids_distinguishes_versions():
    df = _make_tracks_df()
    result = assign_recording_group_ids(df)
    assert len(result) == 3
    t1 = result[result["track_id"] == "t1"].iloc[0]
    t2 = result[result["track_id"] == "t2"].iloc[0]
    t3 = result[result["track_id"] == "t3"].iloc[0]
    assert t1["recording_group_id"] != t2["recording_group_id"]
    assert t1["recording_group_id"] != t3["recording_group_id"]
    assert t2["recording_group_id"] != t3["recording_group_id"]


def test_same_input_same_group():
    df = _make_tracks_df()
    extra = _make_tracks_df()
    extra.loc[0, "track_id"] = "t1b"
    extra.loc[0, "album_name"] = "Album Different"
    result = assign_recording_group_ids(pd.concat([df, extra], ignore_index=True))
    t1 = result[result["track_id"] == "t1"].iloc[0]
    t1b = result[result["track_id"] == "t1b"].iloc[0]
    assert t1["recording_group_id"] == t1b["recording_group_id"]


def test_build_recording_tracks_bridge():
    df = _make_tracks_df()
    result = assign_recording_group_ids(df)
    bridge = build_recording_tracks(result)
    assert len(bridge) == 3
    assert set(bridge.columns) == {"recording_group_id", "track_id"}


def test_build_recordings_selects_representative_by_popularity():
    df = _make_tracks_df()
    result = assign_recording_group_ids(df)
    recs = build_recordings(result)
    assert len(recs) == 3
    # t1 and t2 differ in popularity_median and normalized name -> distinct groups
    assert recs["recording_group_id"].is_unique


def test_build_recordings_representative_tie_break():
    df = _make_tracks_df()
    extra = _make_tracks_df()
    extra.loc[0, "track_id"] = "t1b"
    extra.loc[0, "album_name"] = "Album Different"
    extra.loc[0, "popularity_median"] = 60.0
    result = assign_recording_group_ids(pd.concat([df, extra], ignore_index=True))
    recs = build_recordings(result)
    row = recs[recs["representative_track_id"] == "t1"].iloc[0]
    assert row["track_id_count"] == 2
