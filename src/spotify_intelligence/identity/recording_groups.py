from __future__ import annotations

import pandas as pd

from spotify_intelligence.identity.fingerprints import (
    build_exact_fingerprint,
    fingerprint_to_recording_group_id,
)

_RECORDING_REPRESENTATIVE_COLUMNS: tuple[str, ...] = (
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
)


def assign_recording_group_ids(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add recording_group_id to the track catalog using the exact fingerprint."""
    result = tracks.copy()
    fingerprints = result.apply(build_exact_fingerprint, axis=1)
    result["recording_group_id"] = fingerprints.map(fingerprint_to_recording_group_id)
    return result


def build_recording_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Bridge table: one row per (recording_group_id, track_id)."""
    result = tracks[["recording_group_id", "track_id"]].copy()
    result = result.sort_values(["recording_group_id", "track_id"]).reset_index(drop=True)
    return result


def build_recording_genres(
    recording_tracks: pd.DataFrame, track_genres: pd.DataFrame
) -> pd.DataFrame:
    """Genres per recording group: one row per (recording_group_id, track_genre)."""
    merged = recording_tracks.merge(track_genres, on="track_id", how="inner")
    result = (
        merged[["recording_group_id", "track_genre"]]
        .drop_duplicates()
        .sort_values(["recording_group_id", "track_genre"])
        .reset_index(drop=True)
    )
    return result


def build_anomalies(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per track_id with anomaly indicators for reporting."""
    columns = [
        "track_id",
        "recording_group_id",
        "audio_analysis_incomplete",
        "is_short_track",
        "is_long_track",
    ]
    result = tracks[columns].copy()
    result = result.sort_values(["track_id"]).reset_index(drop=True)
    return result


def _select_representative(tracks: pd.DataFrame) -> pd.DataFrame:
    """Highest popularity_median; ties broken by the lexicographically smallest track_id."""
    ordered = tracks.sort_values(["popularity_median", "track_id"], ascending=[False, True])
    representative = ordered.groupby("recording_group_id", as_index=False).first()
    return representative


def build_recordings(
    tracks: pd.DataFrame,
    recording_genres: pd.DataFrame | None = None,
    track_artists: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per recording_group_id using the representative track."""
    representative = _select_representative(tracks)

    result = representative.rename(columns={"track_id": "representative_track_id"}).copy()

    track_id_counts = tracks.groupby("recording_group_id")["track_id"].count()
    result["track_id_count"] = result["recording_group_id"].map(track_id_counts)

    if recording_genres is not None and not recording_genres.empty:
        genre_counts = recording_genres.groupby("recording_group_id")["track_genre"].count()
        result["genre_count"] = result["recording_group_id"].map(genre_counts)
    else:
        result["genre_count"] = 1

    if track_artists is not None and not track_artists.empty:
        recording_tracks = build_recording_tracks(tracks)
        track_artist_groups = track_artists.merge(recording_tracks, on="track_id", how="inner")
        artist_counts = track_artist_groups.groupby("recording_group_id")["artist"].nunique()
        result["artist_count"] = result["recording_group_id"].map(artist_counts)
    else:
        result["artist_count"] = 1

    result = result[list(_RECORDING_REPRESENTATIVE_COLUMNS)]
    result = result.sort_values("recording_group_id").reset_index(drop=True)
    return result
