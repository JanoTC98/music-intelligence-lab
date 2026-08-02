from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spotify_intelligence.analysis.distributions import _present_features


def feature_correlation(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pearson correlation matrix over the given audio features."""
    cols = _present_features(df, features)
    return df[cols].corr()


def high_correlation_pairs(
    corr: pd.DataFrame,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Pairs whose absolute correlation is at least ``threshold``."""
    rows: list[dict[str, object]] = []
    columns = list(corr.columns)
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            value = corr.loc[feature_a, feature_b]
            if abs(value) >= threshold:
                rows.append({"feature_a": feature_a, "feature_b": feature_b, "correlation": value})
    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
    result = pd.DataFrame(rows)
    return result.sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    title: str = "Matriz de correlación entre características",
) -> plt.Figure:
    """Heatmap with value annotations."""
    features = list(corr.columns)
    values = corr.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(features, fontsize=9)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pairwise_relationship(
    df: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    *,
    title: str = "",
) -> plt.Figure:
    """Scatter with a linear trend line for two features."""
    x = df[feature_a].to_numpy(dtype=float)
    y = df[feature_b].to_numpy(dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, s=4, alpha=0.15, color="#4C72B0")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xs, slope * xs + intercept, color="#C44E52", linewidth=2)
    ax.set_xlabel(feature_a)
    ax.set_ylabel(feature_b)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
