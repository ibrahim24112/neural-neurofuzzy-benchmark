from __future__ import annotations

import numpy as np

from neural_fuzzy_benchmark.models.kmeans import KMeans
from neural_fuzzy_benchmark.models.mlp import one_hot


class RbfTskNeuroFuzzyClassifier:
    def __init__(self, n_rules: int, sigma_scale: float, ridge_penalty: float, seed: int) -> None:
        self.n_rules = n_rules
        self.sigma_scale = sigma_scale
        self.ridge_penalty = ridge_penalty
        self.seed = seed
        self.centers_: np.ndarray | None = None
        self.sigmas_: np.ndarray | None = None
        self.consequent_weights_: np.ndarray | None = None
        self.n_classes_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "RbfTskNeuroFuzzyClassifier":
        self.n_classes_ = n_classes
        kmeans = KMeans(self.n_rules, seed=self.seed, max_iter=120).fit(X)
        if kmeans.centers_ is None:
            raise RuntimeError("K-means failed to create neuro-fuzzy rule centers.")
        self.centers_ = kmeans.centers_
        self.sigmas_ = self._estimate_sigmas(self.centers_)
        features = self._rule_activations(X)
        targets = one_hot(y, n_classes)
        penalty = np.eye(features.shape[1])
        penalty[-1, -1] = 0.0
        self.consequent_weights_ = np.linalg.pinv(features.T @ features + self.ridge_penalty * penalty) @ features.T @ targets
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.consequent_weights_ is None:
            raise RuntimeError("The classifier must be fitted before prediction.")
        return self._rule_activations(X) @ self.consequent_weights_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)

    def _rule_activations(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.sigmas_ is None:
            raise RuntimeError("Rule centers and sigmas are not available.")
        squared_distances = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        activations = np.exp(-squared_distances / (2.0 * self.sigmas_[None, :] ** 2))
        activation_sum = activations.sum(axis=1, keepdims=True)
        normalized = np.divide(activations, activation_sum, out=np.zeros_like(activations), where=activation_sum > 1e-12)
        return np.hstack([normalized, np.ones((len(X), 1))])

    def _estimate_sigmas(self, centers: np.ndarray) -> np.ndarray:
        if len(centers) == 1:
            return np.array([1.0])
        distances = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        distances[distances == 0.0] = np.inf
        nearest = distances.min(axis=1)
        finite = nearest[np.isfinite(nearest)]
        nearest[~np.isfinite(nearest)] = np.median(finite) if len(finite) else 1.0
        nearest = np.where(nearest < 1e-3, 1.0, nearest)
        return nearest * self.sigma_scale + 1e-6
