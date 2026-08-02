from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from spotify_intelligence.analysis.distributions import _present_features


def multi_label_stats(recording_genres: pd.DataFrame) -> pd.DataFrame:
    """Label count summary at recording-group level."""
    per_recording = recording_genres.groupby("recording_group_id")["track_genre"].nunique()
    total = int(len(per_recording))
    single = int((per_recording == 1).sum())
    return pd.DataFrame(
        {
            "metric": [
                "total_recordings",
                "single_label",
                "multi_label",
                "max_labels",
                "mean_labels",
            ],
            "value": [
                total,
                single,
                total - single,
                int(per_recording.max()),
                round(float(per_recording.mean()), 3),
            ],
        }
    )


def genre_recording_counts(recording_genres: pd.DataFrame) -> pd.DataFrame:
    """Number of recordings per genre and share over the catalog."""
    total = int(recording_genres["recording_group_id"].nunique())
    counts = recording_genres["track_genre"].value_counts().rename("recordings")
    result = counts.to_frame()
    result["share_of_recordings"] = result["recordings"] / total
    return result


def cooccurrence_matrix(recording_genres: pd.DataFrame) -> pd.DataFrame:
    """Genre x genre matrix: number of recordings sharing both labels."""
    binary = (
        recording_genres.assign(_value=1)
        .pivot(index="recording_group_id", columns="track_genre", values="_value")
        .fillna(0)
        .astype(int)
    )
    return binary.T @ binary


def top_overlap_pairs(cooc: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Most shared genre pairs by number of co-occurring recordings."""
    genres = list(cooc.columns)
    rows: list[dict[str, object]] = []
    for i, genre_a in enumerate(genres):
        for genre_b in genres[i + 1 :]:
            value = int(cooc.loc[genre_a, genre_b])
            if value > 0:
                rows.append({"genre_a": genre_a, "genre_b": genre_b, "shared_recordings": value})
    if not rows:
        return pd.DataFrame(columns=["genre_a", "genre_b", "shared_recordings"])
    result = pd.DataFrame(rows)
    return (
        result.sort_values("shared_recordings", ascending=False).head(top_n).reset_index(drop=True)
    )


def full_overlap_pairs(cooc: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Pairs where every recording of the smaller genre also has the other label.

    A pair (a, b) is reported when ``shared == min(recordings_a, recordings_b)``,
    meaning the co-occurrence saturates the smaller genre.
    """
    genres = list(cooc.columns)
    rows: list[dict[str, object]] = []
    for i, genre_a in enumerate(genres):
        for genre_b in genres[i + 1 :]:
            shared = int(cooc.loc[genre_a, genre_b])
            cap = min(
                int(counts.loc[genre_a, "recordings"]), int(counts.loc[genre_b, "recordings"])
            )
            if shared > 0 and shared == cap:
                rows.append(
                    {
                        "genre_a": genre_a,
                        "genre_b": genre_b,
                        "shared_recordings": shared,
                        "cap_recordings": cap,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["genre_a", "genre_b", "shared_recordings", "cap_recordings"])
    result = pd.DataFrame(rows)
    return result.sort_values("shared_recordings", ascending=False).reset_index(drop=True)


def genre_acoustic_profiles(
    recordings: pd.DataFrame,
    recording_genres: pd.DataFrame,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean audio features per genre (rows = genres)."""
    cols = _present_features(recordings, features)
    merged = recordings[["recording_group_id", *cols]].merge(
        recording_genres, on="recording_group_id", how="inner"
    )
    return merged.groupby("track_genre")[cols].mean()


def similar_genre_profiles(
    profiles: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Most similar genre pairs by cosine similarity of scaled mean profiles."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles.to_numpy(dtype=float))
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    unit = scaled / np.maximum(norms, 1e-12)
    similarity = pd.DataFrame(
        unit @ unit.T, index=list(profiles.index), columns=list(profiles.index)
    )

    genres = list(profiles.index)
    rows: list[dict[str, object]] = []
    for i, genre_a in enumerate(genres):
        for genre_b in genres[i + 1 :]:
            rows.append(
                {
                    "genre_a": genre_a,
                    "genre_b": genre_b,
                    "profile_cosine": float(similarity.loc[genre_a, genre_b]),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values("profile_cosine", ascending=False).head(top_n).reset_index(drop=True)


def plot_cooccurrence_heatmap(
    cooc: pd.DataFrame,
    *,
    max_genres: int = 25,
    title: str = "Co-ocurrencia entre géneros (subconjunto)",
) -> plt.Figure:
    """Heatmap of co-occurrence restricted to the most connected genres."""
    row_sums = cooc.sum(axis=1).sort_values(ascending=False)
    selected = list(row_sums.head(max_genres).index)
    sub = cooc.loc[selected, selected]
    values = sub.to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(selected)))
    ax.set_yticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=90, fontsize=7)
    ax.set_yticklabels(selected, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6, label="grabaciones compartidas")
    ax.set_title(title)
    fig.tight_layout()
    return fig
