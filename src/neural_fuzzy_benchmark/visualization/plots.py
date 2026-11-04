from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SOFT_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "soft_blue_gradient",
    ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6"],
)


def plot_metric_by_dataset(summary: pd.DataFrame, metric: str = "f1_macro") -> None:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    pivot_mean = summary.pivot(index="dataset", columns="algorithm", values=mean_column)
    pivot_std = summary.pivot(index="dataset", columns="algorithm", values=std_column)
    axis = pivot_mean.plot(kind="bar", yerr=pivot_std, figsize=(12, 5), capsize=3, rot=0)
    axis.set_title(f"{metric.replace('_', ' ').title()} by Dataset and Algorithm")
    axis.set_xlabel("Dataset")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(summary: pd.DataFrame, metric: str = "f1_macro") -> None:
    mean_column = f"{metric}_mean"
    pivot = summary.pivot(index="dataset", columns="algorithm", values=mean_column)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap=SOFT_BLUE_CMAP, vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_xticklabels(pivot.columns, rotation=30, ha="right")
    axis.set_yticklabels(pivot.index)
    axis.set_title(f"Benchmark Heatmap - {metric.replace('_', ' ').title()}")
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            axis.text(column_index, row_index, f"{pivot.iloc[row_index, column_index]:.3f}", ha="center", va="center", color="#102A43")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_from_results(results: pd.DataFrame, dataset: str, algorithm: str, run: int = 1) -> None:
    selected = results[(results["dataset"] == dataset) & (results["algorithm"] == algorithm) & (results["run"] == run)]
    if selected.empty:
        raise ValueError("The selected dataset, algorithm and run were not found.")
    row = selected.iloc[0]
    matrix = _to_matrix(row["confusion_matrix"])
    class_names = row.get("class_names", None)
    if isinstance(class_names, str):
        class_names = ast.literal_eval(class_names)
    plot_confusion_matrix(matrix, class_names=class_names, title=f"Confusion Matrix - {dataset} - {algorithm} - Run {run}")


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str] | tuple[str, ...] | None = None, title: str = "Confusion Matrix") -> None:
    labels = list(class_names) if class_names is not None else [str(index) for index in range(matrix.shape[0])]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, aspect="auto", cmap=SOFT_BLUE_CMAP)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticklabels(labels)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center", color="#102A43")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def _to_matrix(value: object) -> np.ndarray:
    if isinstance(value, str):
        return np.array(ast.literal_eval(value))
    return np.array(value)
