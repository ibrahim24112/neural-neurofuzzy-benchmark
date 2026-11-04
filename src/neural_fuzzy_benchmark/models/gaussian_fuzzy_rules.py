from __future__ import annotations

import numpy as np

from neural_fuzzy_benchmark.models.kmeans import KMeans


class GaussianFuzzyRulesClassifier:
    def __init__(self, rules_per_class: int, sigma_scale: float, seed: int) -> None:
        self.rules_per_class = rules_per_class
        self.sigma_scale = sigma_scale
        self.seed = seed
        self.centers_: np.ndarray | None = None
        self.sigmas_: np.ndarray | None = None
        self.rule_classes_: np.ndarray | None = None
        self.n_classes_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "GaussianFuzzyRulesClassifier":
        centers = []
        sigmas = []
        rule_classes = []
        self.n_classes_ = n_classes

        for class_index in range(n_classes):
            class_features = X[y == class_index]
            cluster_model = KMeans(self.rules_per_class, seed=self.seed + class_index * 37, max_iter=100).fit(class_features)
            if cluster_model.centers_ is None or cluster_model.labels_ is None:
                raise RuntimeError("K-means failed to create fuzzy rule centers.")
            for rule_index, center in enumerate(cluster_model.centers_):
                members = class_features[cluster_model.labels_ == rule_index]
                radius = self._rule_radius(class_features, members, center)
                centers.append(center)
                sigmas.append(radius * self.sigma_scale + 1e-6)
                rule_classes.append(class_index)

        self.centers_ = np.vstack(centers)
        self.sigmas_ = np.array(sigmas, dtype=float)
        self.rule_classes_ = np.array(rule_classes, dtype=int)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.sigmas_ is None or self.rule_classes_ is None or self.n_classes_ is None:
            raise RuntimeError("The classifier must be fitted before prediction.")
        squared_distances = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        strengths = np.exp(-squared_distances / (2.0 * self.sigmas_[None, :] ** 2))
        scores = np.zeros((len(X), self.n_classes_), dtype=float)
        for rule_index, class_index in enumerate(self.rule_classes_):
            scores[:, class_index] += strengths[:, rule_index]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_scores(X), axis=1)

    @staticmethod
    def _rule_radius(class_features: np.ndarray, members: np.ndarray, center: np.ndarray) -> float:
        reference = members if len(members) > 1 else class_features
        distances = np.sqrt(((reference - center) ** 2).sum(axis=1))
        radius = float(distances.mean())
        return radius if np.isfinite(radius) and radius >= 1e-3 else 1.0
