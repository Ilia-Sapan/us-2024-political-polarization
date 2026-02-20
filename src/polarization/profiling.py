"""Cluster profiling and interpretability helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def _column_mean_std(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean and std for dense or sparse matrices."""

    if sparse.issparse(matrix):
        matrix_csr = matrix.tocsr()
        mean = np.asarray(matrix_csr.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix_csr.multiply(matrix_csr).mean(axis=0)).ravel()
        variance = np.maximum(mean_sq - mean**2, 1e-12)
        std = np.sqrt(variance)
        return mean, std

    dense = np.asarray(matrix)
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean, std


def cluster_size_table(labels: np.ndarray) -> pd.DataFrame:
    """Build a cluster size summary table."""

    counts = pd.Series(labels).value_counts().sort_index()
    frame = counts.rename_axis("cluster").reset_index(name="n")
    frame["pct"] = frame["n"] / frame["n"].sum()
    return frame


def compute_feature_contrasts(
    matrix: Any,
    labels: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute top positive and negative standardized feature differences by cluster."""

    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature name length ({len(feature_names)}) does not match matrix width ({matrix.shape[1]})."
        )
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    overall_mean, overall_std = _column_mean_std(matrix)
    results: list[dict[str, Any]] = []

    for cluster_id in sorted(pd.Series(labels).dropna().unique().tolist()):
        cluster_mask = labels == cluster_id
        cluster_matrix = matrix[cluster_mask]
        cluster_mean, _ = _column_mean_std(cluster_matrix)
        standardized_diff = (cluster_mean - overall_mean) / (overall_std + 1e-9)

        top_positive_idx = np.argsort(standardized_diff)[-top_n:][::-1]
        top_negative_idx = np.argsort(standardized_diff)[:top_n]

        for idx in top_positive_idx:
            results.append(
                {
                    "cluster": int(cluster_id),
                    "direction": "positive",
                    "feature": feature_names[idx],
                    "std_diff": float(standardized_diff[idx]),
                    "cluster_mean": float(cluster_mean[idx]),
                    "overall_mean": float(overall_mean[idx]),
                }
            )

        for idx in top_negative_idx:
            results.append(
                {
                    "cluster": int(cluster_id),
                    "direction": "negative",
                    "feature": feature_names[idx],
                    "std_diff": float(standardized_diff[idx]),
                    "cluster_mean": float(cluster_mean[idx]),
                    "overall_mean": float(overall_mean[idx]),
                }
            )

    return pd.DataFrame(results).sort_values(
        by=["cluster", "direction", "std_diff"], ascending=[True, True, False]
    )


def build_cluster_profile_table(
    dataframe: pd.DataFrame, labels: np.ndarray, demographic_columns: Sequence[str]
) -> pd.DataFrame:
    """Create a compact demographic profile table per cluster."""

    prof = dataframe.copy()
    prof["cluster"] = labels

    output = cluster_size_table(labels)
    grouped = prof.groupby("cluster", dropna=False)

    for column in demographic_columns:
        if column not in prof.columns:
            continue

        if pd.api.types.is_numeric_dtype(prof[column]):
            summary = grouped[column].mean().rename(f"{column}_mean").reset_index()
            output = output.merge(summary, on="cluster", how="left")
        else:
            mode_values = grouped[column].apply(
                lambda s: (
                    np.nan
                    if s.dropna().empty
                    else s.dropna().astype(str).value_counts(normalize=True).index[0]
                )
            ).rename(f"{column}_mode")
            mode_share = grouped[column].apply(
                lambda s: (
                    np.nan
                    if s.dropna().empty
                    else float(s.dropna().astype(str).value_counts(normalize=True).iloc[0])
                )
            ).rename(f"{column}_mode_pct")
            merged = pd.concat([mode_values, mode_share], axis=1).reset_index()
            output = output.merge(merged, on="cluster", how="left")

    return output.sort_values("cluster").reset_index(drop=True)

