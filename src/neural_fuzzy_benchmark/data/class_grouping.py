from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassGroupingPolicy:
    enabled: bool = True
    max_classes: int = 3
    min_class_share: float = 0.05
    min_class_count: int = 15


@dataclass(frozen=True)
class ClassGrouping:
    applied: bool
    mapping: dict[int, int]
    grouped_class_names: tuple[str, ...]
    reason: str


def audit_class_distribution(y: np.ndarray, class_names: tuple[str, ...], policy: ClassGroupingPolicy) -> pd.DataFrame:
    counts = pd.Series(y).value_counts().sort_index()
    total = counts.sum()
    rows = []
    for class_index, count in counts.items():
        share = float(count / total)
        rows.append(
            {
                "class": class_names[int(class_index)],
                "samples": int(count),
                "share": share,
                "below_min_share": bool(share < policy.min_class_share),
                "below_min_count": bool(count < policy.min_class_count),
            }
        )
    return pd.DataFrame(rows)


def fit_class_grouping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: tuple[str, ...],
    policy: ClassGroupingPolicy,
) -> ClassGrouping:
    n_classes = len(class_names)
    if not policy.enabled:
        return _identity_grouping(class_names, "Class grouping disabled.")

    counts = pd.Series(y_train).value_counts().sort_index()
    total = counts.sum()
    low_representation = counts[(counts < policy.min_class_count) | ((counts / total) < policy.min_class_share)]
    if n_classes <= policy.max_classes and low_representation.empty:
        return _identity_grouping(class_names, "No grouping required by the selected policy.")

    major_classes = counts.sort_values(ascending=False).head(policy.max_classes).index.to_list()
    mapping = {int(class_index): new_index for new_index, class_index in enumerate(sorted(major_classes))}
    centroids = {
        int(class_index): X_train[y_train == class_index].mean(axis=0)
        for class_index in major_classes
    }

    for class_index in range(n_classes):
        if class_index in mapping:
            continue
        class_centroid = X_train[y_train == class_index].mean(axis=0)
        closest_major_class = min(
            major_classes,
            key=lambda major_class: float(np.linalg.norm(class_centroid - centroids[int(major_class)])),
        )
        mapping[class_index] = mapping[int(closest_major_class)]

    grouped_names = _grouped_names(mapping, class_names)
    return ClassGrouping(True, mapping, grouped_names, "Classes grouped by centroid proximity in the training set.")


def apply_class_grouping(y: np.ndarray, grouping: ClassGrouping) -> np.ndarray:
    return np.array([grouping.mapping[int(value)] for value in y], dtype=int)


def _identity_grouping(class_names: tuple[str, ...], reason: str) -> ClassGrouping:
    return ClassGrouping(False, {index: index for index in range(len(class_names))}, class_names, reason)


def _grouped_names(mapping: dict[int, int], class_names: tuple[str, ...]) -> tuple[str, ...]:
    grouped: dict[int, list[str]] = {}
    for original_index, grouped_index in mapping.items():
        grouped.setdefault(grouped_index, []).append(class_names[original_index])
    return tuple(" + ".join(grouped[index]) for index in sorted(grouped))
