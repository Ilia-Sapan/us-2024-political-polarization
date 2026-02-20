"""Static visualizations saved to reports/figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_embedding(
    embedding_frame: pd.DataFrame,
    output_path: Path,
    x_col: str = "emb_1",
    y_col: str = "emb_2",
    cluster_col: str = "cluster",
) -> None:
    """Save a 2D embedding scatter plot colored by cluster."""

    missing = [col for col in [x_col, y_col, cluster_col] if col not in embedding_frame.columns]
    if missing:
        raise ValueError(f"Embedding plot missing required columns: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        embedding_frame[x_col],
        embedding_frame[y_col],
        c=embedding_frame[cluster_col],
        cmap="tab20",
        s=12,
        alpha=0.75,
    )
    ax.set_title("Embedding of Respondents Colored by Cluster")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    colorbar.set_label("Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_k_selection(scores: Mapping[int, float], output_path: Path) -> None:
    """Save a silhouette-score vs K line plot."""

    if not scores:
        raise ValueError("K-selection scores are empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(scores.items(), key=lambda item: item[0])
    k_values = [item[0] for item in ordered]
    sil_values = [item[1] for item in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, sil_values, marker="o", linewidth=2)
    best_k = k_values[sil_values.index(max(sil_values))]
    best_score = max(sil_values)
    ax.scatter([best_k], [best_score], color="red", zorder=3, label=f"Best K={best_k}")
    ax.set_title("K Selection via Silhouette Score")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

