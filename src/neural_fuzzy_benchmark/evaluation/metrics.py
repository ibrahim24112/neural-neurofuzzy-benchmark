from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> ClassificationMetrics:
    matrix = compute_confusion_matrix(y_true, y_pred, n_classes)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    precision_values = []
    recall_values = []
    f1_values = []

    for class_index in range(n_classes):
        true_positive = float(matrix[class_index, class_index])
        false_positive = float(matrix[:, class_index].sum() - true_positive)
        false_negative = float(matrix[class_index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=float(np.mean(precision_values)),
        recall_macro=float(np.mean(recall_values)),
        f1_macro=float(np.mean(f1_values)),
        confusion_matrix=matrix,
    )
