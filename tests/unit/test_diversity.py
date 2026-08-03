import numpy as np
import pytest

from spotify_intelligence.recommenders.diversity import mmr_rerank, rerank_with_mmr


def test_mmr_rerank_empty():
    order = mmr_rerank(np.empty(0), np.empty((0, 0)))
    assert order.shape == (0,)


def test_mmr_rerank_returns_permutation():
    rng = np.random.default_rng(0)
    relevance = rng.uniform(0.5, 1.0, size=6)
    similarity = np.abs(rng.normal(0, 0.2, size=(6, 6)))
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 1.0)

    order = mmr_rerank(relevance, similarity, lambda_=0.85)
    assert sorted(order.tolist()) == list(range(6))


def test_mmr_first_item_is_max_relevance():
    relevance = np.array([0.5, 1.0, 0.8])
    similarity = np.ones((3, 3))
    order = mmr_rerank(relevance, similarity, lambda_=0.85)
    assert order[0] == 1


def test_mmr_diversity_breaks_similar_cluster():
    relevance = np.array([1.0, 0.99, 0.95])
    similarity = np.array(
        [
            [1.0, 0.99, 0.1],
            [0.99, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    order = mmr_rerank(relevance, similarity, lambda_=0.85)
    assert order[0] == 0
    assert order[1] == 2


def test_mmr_rerank_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mmr_rerank(np.array([1.0, 2.0]), np.ones((3, 3)))


def test_rerank_with_mmr_single_item():
    rows = [{"similarity": 0.9}]
    assert rerank_with_mmr(rows, np.ones((1, 1))) == rows
