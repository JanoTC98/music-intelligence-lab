import pandas as pd

from spotify_intelligence.recommenders.catalog import search_catalog


def make_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "a1",
                "track_name": "Bones",
                "artists": "Imagine Dragons",
                "album_name": "Bones",
            },
            {
                "track_id": "a2",
                "track_name": "Thunder",
                "artists": "Imagine Dragons",
                "album_name": "Evolve",
            },
            {
                "track_id": "a3",
                "track_name": "Believer",
                "artists": "Imagine Dragons",
                "album_name": "Evolve",
            },
            {"track_id": "a4", "track_name": "Drown", "artists": "milet", "album_name": "eyes"},
            {"track_id": "a5", "track_name": "Mirage", "artists": "Angband", "album_name": "IV"},
            {
                "track_id": "a6",
                "track_name": "Bohemian Rhapsody - Remastered 2011",
                "artists": "Queen",
                "album_name": "A Night at the Opera",
            },
            {
                "track_id": "a7",
                "track_name": "Radioactive",
                "artists": "Imagine Dragons",
                "album_name": "Night Visions",
            },
        ]
    )


def test_artist_query_ranks_artist_matches_first():
    result = search_catalog(make_tracks(), "imagine dragons")
    artists = result["artists"].astype(str).str.casefold()
    boosted = artists.str.contains("imagine dragons", regex=False)
    assert len(result) > 0
    assert boosted.iloc[0]
    assert boosted.tolist() == sorted(boosted.tolist(), reverse=True)
    assert boosted.sum() == 4


def test_artist_query_limits_results():
    result = search_catalog(make_tracks(), "imagine dragons", limit=3)
    assert len(result) == 3
    artists = result["artists"].astype(str).str.casefold()
    assert artists.str.contains("imagine dragons", regex=False).all()


def test_track_name_query_keeps_fuzzy_ordering():
    result = search_catalog(make_tracks(), "thunder imagine")
    assert len(result) >= 1
    assert str(result.iloc[0]["track_name"]) == "Thunder"


def test_blank_query_returns_empty():
    assert search_catalog(make_tracks(), "").empty
    assert search_catalog(make_tracks(), "   ").empty


def test_result_excludes_internal_columns():
    result = search_catalog(make_tracks(), "imagine dragons")
    assert "_artist_match" not in result.columns
    assert "track_id" in result.columns
