from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, seed: int, max_iter: int = 100) -> None:
        self.n_clusters = n_clusters
        self.seed = seed
        self.max_iter = max_iter
        self.centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.seed)
        n_clusters = int(max(1, min(self.n_clusters, len(X))))
        centers = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
        labels = np.zeros(len(X), dtype=int)

        for _ in range(self.max_iter):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            updated_labels = np.argmin(distances, axis=1)
            if np.array_equal(updated_labels, labels):
                break
            labels = updated_labels
            for cluster_index in range(n_clusters):
                members = X[labels == cluster_index]
                centers[cluster_index] = members.mean(axis=0) if len(members) else X[rng.integers(0, len(X))]

        self.centers_ = centers
        self.labels_ = labels
        return self
