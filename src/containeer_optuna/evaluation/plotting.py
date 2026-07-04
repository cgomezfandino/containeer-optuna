"""Visualization utilities for embeddings and cluster structure.

These helpers use matplotlib (already a declared dependency of the framework)
to produce static 2D scatter plots of reduced embeddings and scree plots of
explained variance. They complement (but do not replace) Optuna's interactive
plotly visualizations, which are study-centric (search-space diagnostics)
rather than data-centric (cluster structure).

Functions:
    plot_embedding_2d: scatter a 2D embedding, optionally colored by labels.
    plot_scree: bar chart of explained variance ratios (PCA / TruncatedSVD).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def plot_embedding_2d(
    X_2d: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "2D embedding",
    ax: Any | None = None,
) -> Any:
    """Plot a 2D embedding as a scatter, optionally colored by cluster labels.

    Args:
        X_2d: A (n_samples, 2) array of 2D coordinates.
        labels: Optional (n_samples,) array of cluster labels. When provided,
            points are colored by cluster (noise points labeled -1 are drawn
            as small grey crosses). When None, all points share one color.
        title: Plot title.
        ax: Optional matplotlib Axes to draw on. If None, a new Figure/Axes is
            created.

    Returns:
        The matplotlib :class:`Figure` containing the plot.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X_2d)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X_2d must be (n_samples, 2); got shape {X.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if labels is not None:
        labels_arr = np.asarray(labels)
        unique = sorted(set(labels_arr.tolist()) - {-1})
        cmap = plt.get_cmap("tab10", max(len(unique), 1))

        # Noise points (-1) as small grey crosses.
        noise = labels_arr == -1
        if noise.any():
            ax.scatter(
                X[noise, 0],
                X[noise, 1],
                c="lightgrey",
                marker="x",
                s=20,
                label="noise (-1)",
                alpha=0.6,
            )

        # Each cluster as a colored scatter.
        for i, lab in enumerate(unique):
            mask = labels_arr == lab
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[cmap(i)],
                s=40,
                alpha=0.8,
                label=f"cluster {lab}",
                edgecolors="white",
                linewidths=0.5,
            )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, alpha=0.7, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scree(
    explained_variance_ratio: Sequence[float],
    title: str = "Scree plot",
    ax: Any | None = None,
) -> Any:
    """Bar chart of explained variance ratios (for PCA / TruncatedSVD).

    Useful for choosing ``n_components`` via the elbow rule.

    Args:
        explained_variance_ratio: Sequence of per-component variance fractions.
        title: Plot title.
        ax: Optional matplotlib Axes to draw on.

    Returns:
        The matplotlib :class:`Figure` containing the plot.
    """
    import matplotlib.pyplot as plt

    ratios = np.asarray(explained_variance_ratio, dtype=float).ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    components = np.arange(1, len(ratios) + 1)
    cumulative = np.cumsum(ratios)

    ax.bar(components, ratios, color="indigo", alpha=0.7, label="per-component")
    ax.plot(
        components,
        cumulative,
        "o-",
        color="crimson",
        label="cumulative",
    )
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=0.8, label="95% threshold")
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(title)
    ax.set_xticks(components)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


__all__ = ["plot_embedding_2d", "plot_scree"]
