"""Catalog search and seed resolution for the web application (AGENTS.md §14.6).

The application needs a disambiguated lookup of a track by name and artist
(§18.5). This module searches the processed ``tracks.parquet`` catalog with a
flexible, normalized query and returns candidate rows the user can confirm
before any recommendation is computed.

No model is trained here; the search is deterministic and offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

SEARCH_LIMIT_DEFAULT = 20
SEARCH_SCORE_CUTOFF = 45


def _search_query(value: str) -> str:
    """Return a light normalized search token (casefold, trimmed, collapsed)."""
    return " ".join(str(value).strip().casefold().split())


def _candidate_label(row: pd.Series) -> str:
    """Build a single searchable label from a catalog row."""
    name = str(row.get("track_name", "")).strip()
    artists = str(row.get("artists", "")).strip()
    album = str(row.get("album_name", "")).strip()
    return " ".join(part for part in (name, artists, album) if part).casefold()


def _build_search_frame(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return the track-level frame used for fuzzy search.

    The result preserves the original ``tracks`` columns and adds the search
    label plus a stable join key on ``track_id``.
    """
    frame = tracks.copy()
    if "track_name_normalized" not in frame.columns:
        frame["track_name_normalized"] = frame["track_name"].map(_search_query)
    if "artists_normalized" not in frame.columns:
        frame["artists_normalized"] = frame["artists"].map(_search_query)
    frame["_search_label"] = frame.apply(_candidate_label, axis=1)
    return frame


def search_catalog(
    tracks: pd.DataFrame,
    query: str,
    *,
    limit: int = SEARCH_LIMIT_DEFAULT,
) -> pd.DataFrame:
    """Return up to ``limit`` candidate rows matching ``query``.

    Matching uses ``fuzz.token_sort_ratio`` against a combined label built from
    track name, artists and album, so word order matters less than in an exact
    lookup. Candidates are sorted by score (descending) with a stable tie-break
    on ``track_id``.

    An empty or blank query returns an empty frame.
    """
    text = str(query or "").strip()
    if not text:
        return tracks.head(0)

    query_text = text.casefold()
    frame = _build_search_frame(tracks)
    choices = frame["_search_label"].tolist()

    matches = process.extract(
        query_text,
        choices,
        scorer=fuzz.token_sort_ratio,
        limit=limit,
        score_cutoff=SEARCH_SCORE_CUTOFF,
    )
    if not matches:
        return tracks.head(0)

    positions = [int(idx) for _, score, idx in matches]
    result = frame.iloc[positions].copy()
    result["_search_score"] = [int(score) for _, score, _ in matches]
    # Boost candidates whose artist matches the query so typing an artist name
    # surfaces that artist's catalog before loose fuzzy token matches.
    result["_artist_match"] = (
        result["artists"]
        .astype(str)
        .str.casefold()
        .str.contains(query_text, regex=False)
        .astype(int)
    )
    result = result.sort_values(
        ["_artist_match", "_search_score", "track_id"],
        ascending=[False, False, True],
    )
    result = result.drop_duplicates(subset=["track_id"], keep="first")
    return result.drop(columns=["_artist_match"]).reset_index(drop=True)


def resolve_track(
    tracks: pd.DataFrame,
    track_id: str,
) -> pd.Series:
    """Return the catalog row for a confirmed ``track_id``.

    Raises ``KeyError`` when the identifier is absent so callers can surface a
    clear user-facing message instead of an empty recommendation.
    """
    frame = tracks.copy()
    if "track_id" not in frame.columns:
        raise KeyError("tracks catalog has no track_id column")
    frame = frame.set_index("track_id")
    if track_id not in frame.index:
        raise KeyError(f"track_id not found in catalog: {track_id}")
    return frame.loc[track_id]


def build_display_rows(
    recommendations: pd.DataFrame,
    *,
    genres_by_group: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Normalize recommendation output columns for presentation.

    ``recommendations`` is the DataFrame returned by a recommender. The genres
    mapping comes from ``recording_genres.parquet`` grouped by
    ``recording_group_id``.
    """
    rows = recommendations.copy()
    if "genres" not in rows.columns and genres_by_group:
        rows["genres"] = rows["recording_group_id"].map(genres_by_group)
    if "genres" in rows.columns:
        rows["genres"] = rows["genres"].apply(
            lambda values: ", ".join(values) if isinstance(values, (list, tuple)) else values
        )
    return rows


def genres_by_recording_group(
    recording_genres: pd.DataFrame,
) -> dict[str, list[str]]:
    """Build ``{recording_group_id: [genre, ...]}`` from the bridge table."""
    return {
        str(group): genres
        for group, genres in recording_genres.groupby("recording_group_id", sort=False)[
            "track_genre"
        ]
        .apply(lambda values: [str(v) for v in values])
        .to_dict()
        .items()
    }


def genres_for_recording(
    recording_genres: pd.DataFrame,
    recording_group_id: str,
) -> list[str]:
    """Return the sorted genre labels of a single recording group."""
    rows = recording_genres[
        recording_genres["recording_group_id"].astype(str) == str(recording_group_id)
    ]["track_genre"].tolist()
    return sorted(str(value) for value in rows)


def read_catalog_tracks(processed_dir: str | Path) -> pd.DataFrame:
    """Load ``tracks.parquet`` for the search index."""
    return pd.read_parquet(Path(processed_dir) / "tracks.parquet")


def read_genre_bridge(processed_dir: str | Path) -> pd.DataFrame:
    """Load ``recording_genres.parquet`` for display purposes."""
    return pd.read_parquet(Path(processed_dir) / "recording_genres.parquet")
