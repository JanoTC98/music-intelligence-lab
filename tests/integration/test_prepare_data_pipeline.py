import json

import pandas as pd
import pytest

from spotify_intelligence.data.pipeline import prepare_data

RAW_ROWS = 114000
VALID_TRACK_IDS = 89740
RECORDING_GROUP_COUNT = 83881


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run the real pipeline once against the real dataset."""
    out_dir = tmp_path_factory.mktemp("pipeline")
    manifest = prepare_data(
        dataset_path="data/raw/dataset.csv",
        config_path="configs/data_rules.yaml",
        output_root=str(out_dir),
        generate_candidates=False,
    )
    return {
        "manifest": manifest,
        "output_root": out_dir,
    }


def test_manifest_counts(prepared):
    manifest = prepared["manifest"]
    assert manifest["raw_rows"] == RAW_ROWS
    assert manifest["valid_track_ids"] == VALID_TRACK_IDS
    assert manifest["recording_group_count"] == RECORDING_GROUP_COUNT
    assert len(manifest["dataset_sha256"]) == 64


def test_quarantine_invalid_identity(prepared):
    path = prepared["output_root"] / "data" / "quarantine" / "invalid_identity.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 1


def test_tracks_contract(prepared):
    tracks = pd.read_parquet(prepared["output_root"] / "data" / "processed" / "tracks.parquet")
    assert len(tracks) == VALID_TRACK_IDS
    assert tracks["track_id"].is_unique
    assert not tracks["recording_group_id"].isna().any()
    required = [
        "track_id",
        "track_name",
        "track_name_normalized",
        "artists",
        "artists_normalized",
        "album_name",
        "popularity_min",
        "popularity_max",
        "popularity_median",
        "popularity_observations",
        "duration_ms",
        "duration_min",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
        "audio_analysis_incomplete",
        "is_short_track",
        "is_long_track",
        "recording_group_id",
    ]
    assert all(col in tracks.columns for col in required)


def test_recordings_contract(prepared):
    recordings = pd.read_parquet(
        prepared["output_root"] / "data" / "processed" / "recordings.parquet"
    )
    assert len(recordings) == RECORDING_GROUP_COUNT
    assert recordings["recording_group_id"].is_unique
    assert recordings["track_id_count"].sum() == VALID_TRACK_IDS
    required = [
        "recording_group_id",
        "representative_track_id",
        "track_name",
        "artists",
        "album_name",
        "track_id_count",
        "genre_count",
        "artist_count",
        "popularity_median",
        "duration_ms",
        "explicit",
    ]
    assert all(col in recordings.columns for col in required)


def test_no_orphan_track_ids(prepared):
    tracks = pd.read_parquet(prepared["output_root"] / "data" / "processed" / "tracks.parquet")
    recording_tracks = pd.read_parquet(
        prepared["output_root"] / "data" / "processed" / "recording_tracks.parquet"
    )
    missing = set(tracks["track_id"]) - set(recording_tracks["track_id"])
    assert not missing


def test_genre_catalog_114(prepared):
    catalog = pd.read_parquet(
        prepared["output_root"] / "data" / "processed" / "genre_catalog.parquet"
    )
    assert len(catalog) == 114


def test_anomalies_reference_counts(prepared):
    tracks = pd.read_parquet(prepared["output_root"] / "data" / "processed" / "tracks.parquet")
    assert int(tracks["audio_analysis_incomplete"].sum()) == 157


def test_manifest_file_written(prepared):
    manifest_path = prepared["output_root"] / "data" / "processed" / "prepare_data_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        data = json.load(f)
    assert data["raw_rows"] == RAW_ROWS
