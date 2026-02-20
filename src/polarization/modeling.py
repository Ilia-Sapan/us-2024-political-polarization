"""Modeling functions for clustering and dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score

try:
    import umap  # type: ignore

    UMAP_AVAILABLE = True
except Exception:
    umap = None
    UMAP_AVAILABLE = False

try:
    import hdbscan  # type: ignore

    HDBSCAN_AVAILABLE = True
except Exception:
    hdbscan = None
    HDBSCAN_AVAILABLE = False


@dataclass(frozen=True)
class KSelectionResult:
    """KMeans model-selection output."""

    best_k: int
    scores: dict[int, float]


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding output with method metadata."""

    method: str
    coordinates: np.ndarray
    reducer: Any


def _sample_rows(
    matrix: Any, sample_size: int, random_state: int
) -> tuple[Any, np.ndarray | None]:
    """Optionally sample rows from a matrix to speed up evaluation."""

    n_rows = matrix.shape[0]
    if n_rows <= sample_size:
        return matrix, None

    rng = np.random.default_rng(seed=random_state)
    indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
    return matrix[indices], indices


def select_best_kmeans_k(
    matrix: Any,
    k_min: int = 2,
    k_max: int = 12,
    random_state: int = 42,
    sample_size: int = 10_000,
) -> KSelectionResult:
    """Select K by maximizing silhouette score over a K range."""

    if k_min < 2:
        raise ValueError("k_min must be at least 2.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")

    eval_matrix, _ = _sample_rows(matrix, sample_size=sample_size, random_state=random_state)
    scores: dict[int, float] = {}

    for k_value in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k_value, random_state=random_state, n_init="auto")
        labels = model.fit_predict(eval_matrix)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            scores[k_value] = -1.0
            continue
        score = silhouette_score(eval_matrix, labels, metric="euclidean")
        scores[k_value] = float(score)

    best_k = max(scores, key=scores.get)
    return KSelectionResult(best_k=best_k, scores=scores)


def fit_kmeans(
    matrix: Any, n_clusters: int, random_state: int = 42
) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans on the full matrix and return labels."""

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(matrix)
    return model, labels


def compute_silhouette(
    matrix: Any, labels: np.ndarray, sample_size: int = 10_000, random_state: int = 42
) -> float | None:
    """Compute silhouette score on all data or a row sample."""

    if np.unique(labels).size < 2:
        return None

    eval_matrix, indices = _sample_rows(matrix, sample_size=sample_size, random_state=random_state)
    eval_labels = labels if indices is None else labels[indices]
    return float(silhouette_score(eval_matrix, eval_labels, metric="euclidean"))


def reduce_dimensions(
    matrix: Any,
    n_components: int = 2,
    method: str = "umap",
    random_state: int = 42,
) -> EmbeddingResult:
    """Generate a low-dimensional embedding with UMAP or PCA fallback."""

    if n_components < 2:
        raise ValueError("n_components must be at least 2 for visualization.")

    requested = method.lower().strip()

    if requested == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.0,
            metric="euclidean",
            random_state=random_state,
        )
        coordinates = reducer.fit_transform(matrix)
        return EmbeddingResult(method="umap", coordinates=coordinates, reducer=reducer)

    if sparse.issparse(matrix):
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    else:
        reducer = PCA(n_components=n_components, random_state=random_state)

    coordinates = reducer.fit_transform(matrix)
    return EmbeddingResult(method="pca", coordinates=coordinates, reducer=reducer)


def fit_optional_hdbscan(
    embedding: np.ndarray, min_cluster_size: int = 200
) -> tuple[np.ndarray | None, Any | None]:
    """Run HDBSCAN on embedding space when dependency is available."""

    if not HDBSCAN_AVAILABLE:
        return None, None

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer

