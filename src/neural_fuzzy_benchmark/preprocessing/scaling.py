from __future__ import annotations

import numpy as np


class StandardScalerFromTraining:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerFromTraining":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("The scaler must be fitted before transform.")
        return (X - self.mean_) / self.std_

    def fit_transform_split(self, X_train: np.ndarray, X_validation: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(X_train)
        return self.transform(X_train), self.transform(X_validation), self.transform(X_test)
