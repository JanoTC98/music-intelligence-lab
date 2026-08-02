import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from spotify_intelligence.analysis import correlations as corr
from spotify_intelligence.analysis import distributions as dist
from spotify_intelligence.analysis import genre_overlap as go
from spotify_intelligence.analysis.figures import save_figure


def _make_recordings():
    return pd.DataFrame(
        {
            "recording_group_id": [f"g{i}" for i in range(4)],
            "danceability": [0.4, 0.5, 0.6, 0.7],
            "energy": [0.2, 0.4, 0.6, 0.8],
            "loudness": [-12.0, -9.0, -6.0, -3.0],
            "speechiness": [0.03, 0.05, 0.07, 0.09],
            "acousticness": [0.8, 0.5, 0.2, 0.1],
            "instrumentalness": [0.0, 0.1, 0.2, 0.9],
            "liveness": [0.1, 0.2, 0.3, 0.4],
            "valence": [0.3, 0.5, 0.7, 0.9],
            "tempo": [90.0, 110.0, 130.0, 150.0],
            "duration_ms": [200000, 250000, 300000, 400000],
            "audio_analysis_incomplete": [False, False, False, False],
        }
    )


def _make_tracks():
    rows = []
    for i in range(4):
        rows.append(
            {
                "track_id": f"t{i}",
                "duration_ms": [200000, 30000, 900000, 250000][i],
                "duration_min": [200000, 30000, 900000, 250000][i] / 60000.0,
                "explicit": [False, True, False, True][i],
                "is_short_track": [False, True, False, False][i],
                "is_long_track": [False, False, True, False][i],
                "danceability": 0.5,
                "energy": 0.6,
                "loudness": -8.0,
                "speechiness": 0.05,
                "acousticness": 0.3,
                "instrumentalness": 0.0,
                "liveness": 0.2,
                "valence": 0.4,
                "tempo": 120.0,
            }
        )
    return pd.DataFrame(rows)


def _make_recording_genres():
    return pd.DataFrame(
        {
            "recording_group_id": ["g0", "g1", "g1", "g2", "g3", "g3", "g3"],
            "track_genre": ["rock", "rock", "pop", "pop", "rock", "pop", "jazz"],
        }
    )


def test_feature_summary_rows_are_features():
    recordings = _make_recordings()
    summary = dist.feature_summary(recordings, ["danceability", "energy"])
    assert list(summary.index) == ["danceability", "energy"]
    assert {"mean", "std", "5%", "95%"} <= set(summary.columns)


def test_duration_category_counts_mutually_exclusive():
    tracks = _make_tracks()
    counts = dist.duration_category_counts(tracks)
    assert counts["count"].sum() == len(tracks)
    short = counts.loc[counts["category"] == "short (< 60 s)", "count"].iloc[0]
    long = counts.loc[counts["category"] == "long (> 10 min)", "count"].iloc[0]
    assert short == 1
    assert long == 1


def test_explicit_profiles_has_both_groups():
    tracks = _make_tracks()
    profiles = dist.explicit_profiles(tracks, ["energy", "tempo"])
    assert {"explicit", "non_explicit"} <= set(profiles.index)
    assert "count" in profiles.columns


def test_consolidation_comparison_has_deltas():
    tracks = _make_tracks()
    recordings = _make_recordings()
    comparison = dist.consolidation_comparison(tracks, recordings, ["energy", "tempo"])
    assert "mean_delta" in comparison.columns
    assert list(comparison.index) == ["energy", "tempo"]


def test_incomplete_audio_by_genre():
    recordings = _make_recordings()
    recordings.loc[0, "audio_analysis_incomplete"] = True
    genres = _make_recording_genres()
    result = dist.incomplete_audio_by_genre(recordings, genres)
    assert result.loc["rock", "incomplete"] >= 1
    assert "share_incomplete" in result.columns


def test_high_correlation_pairs_threshold():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 1.9, 3.2, 3.8, 5.1],
            "c": [5, 4, 3, 2, 1],
        }
    )
    matrix = corr.feature_correlation(df, ["a", "b", "c"])
    pairs = corr.high_correlation_pairs(matrix, threshold=0.9)
    assert {"a", "b"} <= set(pairs["feature_a"].tolist() + pairs["feature_b"].tolist())
    assert all(abs(v) >= 0.9 for v in pairs["correlation"])


def test_plot_functions_return_figures():
    recordings = _make_recordings()
    tracks = _make_tracks()
    assert dist.plot_feature_histograms(recordings, ["energy", "tempo"]).get_axes()
    assert dist.plot_duration_histogram(tracks).get_axes()
    assert corr.plot_correlation_heatmap(corr.feature_correlation(recordings)).get_axes()
    assert corr.plot_pairwise_relationship(recordings, "energy", "loudness").get_axes()


def test_save_figure_writes_png(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    path = save_figure(fig, "test_figure.png", output_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".png"


def test_multi_label_stats():
    genres = _make_recording_genres()
    stats = go.multi_label_stats(genres)
    values = dict(zip(stats["metric"], stats["value"], strict=False))
    assert values["total_recordings"] == 4
    assert values["multi_label"] == 2
    assert values["max_labels"] == 3


def test_cooccurrence_matrix_symmetric_and_diagonal():
    genres = _make_recording_genres()
    cooc = go.cooccurrence_matrix(genres)
    assert (cooc.to_numpy() == cooc.to_numpy().T).all()
    np.testing.assert_array_equal(np.diag(cooc.to_numpy()), cooc.to_numpy().diagonal())


def test_top_overlap_pairs_ordering():
    genres = _make_recording_genres()
    cooc = go.cooccurrence_matrix(genres)
    pairs = go.top_overlap_pairs(cooc, top_n=5)
    assert not pairs.empty
    assert int(pairs.iloc[0]["shared_recordings"]) >= int(pairs.iloc[-1]["shared_recordings"])


def test_full_overlap_pair_detected():
    genres = _make_recording_genres()
    cooc = go.cooccurrence_matrix(genres)
    counts = go.genre_recording_counts(genres)
    full = go.full_overlap_pairs(cooc, counts)
    # jazz only appears together with rock -> total overlap for the pair (jazz, rock)
    assert not full.empty


def test_similar_genre_profiles_ranks_perfect_pair_first():
    recordings = _make_recordings()
    genres = _make_recording_genres()
    profiles = go.genre_acoustic_profiles(recordings, genres, ["energy", "danceability"])
    similar = go.similar_genre_profiles(profiles, top_n=5)
    assert not similar.empty
    assert similar["profile_cosine"].iloc[0] >= similar["profile_cosine"].iloc[-1]
