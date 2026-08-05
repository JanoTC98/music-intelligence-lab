from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

AUDIO_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)


def _present_features(df: pd.DataFrame, features: Sequence[str] | None) -> list[str]:
    base = list(features) if features is not None else list(AUDIO_FEATURES)
    return [col for col in base if col in df.columns]


def feature_summary(df: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Descriptive statistics per audio feature (rows = features)."""
    cols = _present_features(df, features)
    return df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def plot_feature_histograms(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    title: str = "",
    bins: int = 60,
    cols: int = 3,
) -> plt.Figure:
    """Grid of histograms for each audio feature."""
    feature_cols = _present_features(df, features)
    rows = (len(feature_cols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for ax, feature in zip(axes.flat, feature_cols, strict=False):
        ax.hist(df[feature].dropna(), bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_title(feature, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(feature_cols) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def duration_category_counts(tracks: pd.DataFrame) -> pd.DataFrame:
    """Counts of short, long and normal tracks at track_id level."""
    total = len(tracks)
    short = int(tracks["is_short_track"].sum())
    long = int(tracks["is_long_track"].sum())
    normal = total - short - long
    rows = [
        {"category": "normal", "count": normal, "share_of_tracks": normal / total},
        {"category": "short (< 60 s)", "count": short, "share_of_tracks": short / total},
        {"category": "long (> 10 min)", "count": long, "share_of_tracks": long / total},
    ]
    return pd.DataFrame(rows)


def plot_duration_histogram(tracks: pd.DataFrame, *, title: str = "") -> plt.Figure:
    """Histogram of track duration in minutes."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(tracks["duration_min"], bins=80, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Duración (min)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title or "Distribución de la duración de pistas")
    fig.tight_layout()
    return fig


def explicit_profiles(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean audio features per explicit / non-explicit group with row counts."""
    cols = _present_features(df, features)
    grouped = df.groupby("explicit")[cols].mean()
    counts = df.groupby("explicit").size()
    result = grouped.assign(count=counts)
    result.index = ["explicit" if bool(value) else "non_explicit" for value in result.index]
    return result


def consolidation_comparison(
    tracks: pd.DataFrame,
    recordings: pd.DataFrame,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean/std per feature at track level vs recording-group level."""
    cols = _present_features(tracks, features)
    track_stats = (
        tracks[cols]
        .describe()
        .T[["mean", "std"]]
        .rename(columns={"mean": "mean_tracks", "std": "std_tracks"})
    )
    recording_stats = (
        recordings[cols]
        .describe()
        .T[["mean", "std"]]
        .rename(columns={"mean": "mean_recordings", "std": "std_recordings"})
    )
    result = track_stats.join(recording_stats)
    result["mean_delta"] = result["mean_recordings"] - result["mean_tracks"]
    return result


def incomplete_audio_by_genre(
    recordings: pd.DataFrame,
    recording_genres: pd.DataFrame,
) -> pd.DataFrame:
    """Audio-incomplete recordings by genre (count and share)."""
    merged = recordings[["recording_group_id", "audio_analysis_incomplete"]].merge(
        recording_genres, on="recording_group_id", how="inner"
    )
    per_genre = merged.groupby("track_genre")["audio_analysis_incomplete"].agg(
        total="count", incomplete="sum"
    )
    per_genre["share_incomplete"] = per_genre["incomplete"] / per_genre["total"]
    return per_genre.sort_values("incomplete", ascending=False)  # type: ignore[no-any-return, call-overload]
