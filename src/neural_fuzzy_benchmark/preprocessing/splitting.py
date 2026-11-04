from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class StratifiedSplitter:
    def __init__(self, train_fraction: float = 0.60, validation_fraction: float = 0.20) -> None:
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction

    def split(self, y: np.ndarray, seed: int) -> DataSplit:
        rng = np.random.default_rng(seed)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        test_indices: list[int] = []

        for class_value in np.unique(y):
            indices = np.flatnonzero(y == class_value)
            rng.shuffle(indices)
            n_samples = len(indices)
            n_train = max(1, int(round(self.train_fraction * n_samples)))
            n_validation = max(1, int(round(self.validation_fraction * n_samples)))
            if n_train + n_validation >= n_samples:
                n_validation = max(1, n_samples - n_train - 1)
            train_indices.extend(indices[:n_train])
            validation_indices.extend(indices[n_train : n_train + n_validation])
            test_indices.extend(indices[n_train + n_validation :])

        for indices in (train_indices, validation_indices, test_indices):
            rng.shuffle(indices)

        return DataSplit(
            train=np.array(train_indices, dtype=int),
            validation=np.array(validation_indices, dtype=int),
            test=np.array(test_indices, dtype=int),
        )
