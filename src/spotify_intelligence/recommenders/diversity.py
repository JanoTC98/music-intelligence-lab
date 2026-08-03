"""Optional MMR diversity reranking (AGENTS.md §15.9).

Score = lambda * relevance - (1 - lambda) * similarity_to_selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mmr_rerank(
    relevance: np.ndarray,
    similarity: np.ndarray,
    *,
    lambda_: float = 0.85,
) -> np.ndarray:
    """Greedy MMR reranking over ``relevance`` and pairwise ``similarity``.

    ``relevance`` is a 1D array of per-item relevance scores. ``similarity`` is a
    symmetric 2D matrix of pairwise similarities. Returns the index order that
    maximizes marginal relevance.
    """
    n = int(relevance.shape[0])
    if n == 0:
        return np.empty(0, dtype=int)
    if similarity.shape != (n, n):
        raise ValueError(f"similarity must be ({n}, {n}), got {similarity.shape}")

    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(n):
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            if not selected:
                score = lambda_ * float(relevance[idx])
            else:
                max_sim = max(float(similarity[idx, s]) for s in selected)
                score = lambda_ * float(relevance[idx]) - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return np.asarray(selected, dtype=int)


def rerank_with_mmr(
    results: list[dict[str, Any]],
    pairwise_similarity: np.ndarray,
    *,
    lambda_: float = 0.85,
) -> list[dict[str, Any]]:
    """Rerank a list of result rows by MMR using the embedded similarity score."""
    if len(results) <= 1:
        return results
    relevance = np.asarray([float(r["similarity"]) for r in results])
    order = mmr_rerank(relevance, pairwise_similarity, lambda_=lambda_)
    return [results[i] for i in order]
