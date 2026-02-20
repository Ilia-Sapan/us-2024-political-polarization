"""Tests for clustering and embedding functions."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs

from polarization.modeling import (
    compute_silhouette,
    fit_kmeans,
    reduce_dimensions,
    select_best_kmeans_k,
)


def test_select_best_kmeans_k_returns_valid_range() -> None:
    """Selected K should lie inside the provided bounds."""

    matrix, _ = make_blobs(
        n_samples=300,
        n_features=6,
        centers=3,
        cluster_std=0.7,
        random_state=42,
    )
    result = select_best_kmeans_k(matrix, k_min=2, k_max=5, random_state=42, sample_size=300)
    assert 2 <= result.best_k <= 5
    assert set(result.scores.keys()) == {2, 3, 4, 5}


def test_reduce_dimensions_pca_output_shape() -> None:
    """PCA fallback should produce the configured number of dimensions."""

    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(100, 20))
    embedding = reduce_dimensions(matrix, n_components=2, method="pca", random_state=42)
    assert embedding.coordinates.shape == (100, 2)
    assert embedding.method == "pca"


def test_compute_silhouette_on_kmeans_labels() -> None:
    """Silhouette should be finite for separable clusters."""

    matrix, _ = make_blobs(
        n_samples=250,
        n_features=5,
        centers=4,
        cluster_std=0.9,
        random_state=7,
    )
    _, labels = fit_kmeans(matrix, n_clusters=4, random_state=7)
    score = compute_silhouette(matrix, labels, sample_size=250, random_state=7)
    assert score is not None
    assert -1.0 <= score <= 1.0

