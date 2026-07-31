from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_FLEXIBLE_PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")

RECOMMENDER_FEATURES: tuple[str, ...] = (
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


def flexible_title_key(value: str) -> str:
    """More flexible title normalization for candidate discovery only."""
    normalized = _FLEXIBLE_PUNCTUATION_RE.sub(" ", str(value))
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip().casefold()
    return normalized


def _deduplicate_symmetric_pairs(rows: list[dict[str, object]]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    left = result["recording_group_id_a"].to_numpy()
    right = result["recording_group_id_b"].to_numpy()
    pair_matrix = np.stack([left, right], axis=1)
    pair_matrix.sort(axis=1)
    result = result.assign(_key_a=pair_matrix[:, 0], _key_b=pair_matrix[:, 1])
    result = result.drop_duplicates(subset=["_key_a", "_key_b"])
    result = result.drop(columns=["_key_a", "_key_b"])
    return result


def _textual_candidates(
    recordings: pd.DataFrame,
    duration_tolerance_ms: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    work = recordings.copy()
    work["_title_key"] = work["track_name"].map(
        lambda value: flexible_title_key(value) if pd.notna(value) else ""
    )

    for _key, group in work.groupby("_title_key", sort=False):
        if len(group) < 2:
            continue

        sorted_group = group.sort_values("recording_group_id").reset_index(drop=True)
        for i in range(len(sorted_group)):
            base = sorted_group.iloc[i]
            for j in range(i + 1, len(sorted_group)):
                candidate = sorted_group.iloc[j]
                duration_diff = abs(int(base["duration_ms"]) - int(candidate["duration_ms"]))
                if duration_diff > duration_tolerance_ms:
                    continue
                title_similarity = fuzz.ratio(str(base["track_name"]), str(candidate["track_name"]))
                rows.append(
                    {
                        "recording_group_id_a": base["recording_group_id"],
                        "recording_group_id_b": candidate["recording_group_id"],
                        "track_name": base["track_name"],
                        "artists_a": base["artists"],
                        "artists_b": candidate["artists"],
                        "duration_diff_ms": duration_diff,
                        "title_similarity": title_similarity,
                        "acoustic_similarity": None,
                        "evidence": "textual",
                    }
                )
    return rows


def _acoustic_candidates(
    recordings: pd.DataFrame,
    *,
    top_k: int,
    duration_tolerance_ms: int,
    similarity_threshold: float,
) -> list[dict[str, object]]:
    eligible = recordings[~recordings["audio_analysis_incomplete"]].copy()
    if len(eligible) < 2:
        return []

    feature_columns = [col for col in RECOMMENDER_FEATURES if col in eligible.columns]
    if not feature_columns:
        return []

    matrix = eligible[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    unit = scaled / np.maximum(norms, 1e-12)

    nn = NearestNeighbors(
        n_neighbors=min(top_k + 1, len(eligible)),
        metric="euclidean",
        algorithm="kd_tree",
    )
    nn.fit(unit)
    distances, indices = nn.kneighbors(unit)

    group_ids = eligible["recording_group_id"].to_numpy()
    durations = eligible["duration_ms"].to_numpy()
    track_names = eligible["track_name"].to_numpy()
    artists = eligible["artists"].to_numpy()

    rows: list[dict[str, object]] = []
    append = rows.append
    for row_idx in range(len(eligible)):
        base_id = group_ids[row_idx]
        base_duration = int(durations[row_idx])
        for neighbor_idx, euclidean_dist in zip(indices[row_idx], distances[row_idx], strict=False):
            if neighbor_idx == row_idx:
                continue
            similarity = float(1.0 - 0.5 * euclidean_dist**2)
            if similarity < similarity_threshold:
                continue
            duration_diff = abs(base_duration - int(durations[neighbor_idx]))
            if duration_diff > duration_tolerance_ms:
                continue
            append(
                {
                    "recording_group_id_a": base_id,
                    "recording_group_id_b": group_ids[neighbor_idx],
                    "track_name": track_names[row_idx],
                    "artists_a": artists[row_idx],
                    "artists_b": artists[neighbor_idx],
                    "duration_diff_ms": duration_diff,
                    "title_similarity": None,
                    "acoustic_similarity": round(similarity, 6),
                    "evidence": "acoustic",
                }
            )
    return rows


def build_near_duplicate_candidates(
    recordings: pd.DataFrame,
    *,
    top_k: int = 10,
    duration_tolerance_ms: int = 30000,
    acoustic_similarity_threshold: float = 0.98,
) -> pd.DataFrame:
    """Report candidate near duplicates WITHOUT merging them.

    Two independent signals are combined:

    - Textual: recordings sharing a flexible normalized title key are compared
      with RapidFuzz; pairs within the duration tolerance are kept.
    - Acoustic: brute-force cosine Nearest Neighbors on scaled audio features;
      pairs above the similarity threshold within the duration tolerance are kept.

    This module never merges recordings automatically.
    """
    textual = _textual_candidates(recordings, duration_tolerance_ms)
    acoustic = _acoustic_candidates(
        recordings,
        top_k=top_k,
        duration_tolerance_ms=duration_tolerance_ms,
        similarity_threshold=acoustic_similarity_threshold,
    )

    combined = _deduplicate_symmetric_pairs(textual + acoustic)

    columns = [
        "recording_group_id_a",
        "recording_group_id_b",
        "track_name",
        "artists_a",
        "artists_b",
        "duration_diff_ms",
        "title_similarity",
        "acoustic_similarity",
        "evidence",
    ]
    if combined.empty:
        return pd.DataFrame(columns=columns)

    combined = combined[columns]
    combined = combined.sort_values(
        ["track_name", "recording_group_id_a", "recording_group_id_b"]
    ).reset_index(drop=True)
    return combined
