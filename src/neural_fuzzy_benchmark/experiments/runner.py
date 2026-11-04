from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from neural_fuzzy_benchmark.data.class_grouping import ClassGroupingPolicy, apply_class_grouping, fit_class_grouping
from neural_fuzzy_benchmark.data.loaders import LoadedDataset
from neural_fuzzy_benchmark.evaluation.metrics import compute_classification_metrics
from neural_fuzzy_benchmark.experiments.model_factory import create_classifier
from neural_fuzzy_benchmark.experiments.parameter_grids import build_parameter_grids
from neural_fuzzy_benchmark.preprocessing.scaling import StandardScalerFromTraining
from neural_fuzzy_benchmark.preprocessing.splitting import StratifiedSplitter


@dataclass(frozen=True)
class ExperimentSettings:
    n_runs: int = 21
    base_seed: int = 2026016
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    class_grouping_policy: ClassGroupingPolicy = ClassGroupingPolicy()


def run_experiment_suite(datasets: dict[str, LoadedDataset], settings: ExperimentSettings) -> pd.DataFrame:
    parameter_grids = build_parameter_grids()
    rows = []
    for dataset in datasets.values():
        rows.extend(_run_dataset(dataset, parameter_grids, settings))
    return pd.DataFrame(rows)


def summarize_metrics(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "elapsed_seconds"]
    summary = results.groupby(["dataset", "algorithm"], as_index=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    return summary


def _run_dataset(dataset: LoadedDataset, parameter_grids: dict[str, list[dict[str, object]]], settings: ExperimentSettings) -> list[dict[str, object]]:
    rows = []
    splitter = StratifiedSplitter(settings.train_fraction, settings.validation_fraction)

    for run_index in range(settings.n_runs):
        seed = settings.base_seed + run_index
        split = splitter.split(dataset.target, seed)
        X_train, X_validation, X_test = dataset.features[split.train], dataset.features[split.validation], dataset.features[split.test]
        y_train_raw, y_validation_raw, y_test_raw = dataset.target[split.train], dataset.target[split.validation], dataset.target[split.test]
        X_train, X_validation, X_test = StandardScalerFromTraining().fit_transform_split(X_train, X_validation, X_test)
        grouping = fit_class_grouping(X_train, y_train_raw, dataset.class_names, settings.class_grouping_policy)
        y_train = apply_class_grouping(y_train_raw, grouping)
        y_validation = apply_class_grouping(y_validation_raw, grouping)
        y_test = apply_class_grouping(y_test_raw, grouping)
        n_classes = len(grouping.grouped_class_names)

        for algorithm, parameter_grid in parameter_grids.items():
            rows.append(
                _tune_and_evaluate(
                    dataset=dataset,
                    algorithm=algorithm,
                    parameter_grid=parameter_grid,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    X_test=X_test,
                    y_test=y_test,
                    n_classes=n_classes,
                    class_names=grouping.grouped_class_names,
                    seed=seed,
                    run=run_index + 1,
                    class_grouping_applied=grouping.applied,
                    class_grouping_reason=grouping.reason,
                )
            )
    return rows


def _tune_and_evaluate(
    dataset: LoadedDataset,
    algorithm: str,
    parameter_grid: list[dict[str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    class_names: tuple[str, ...],
    seed: int,
    run: int,
    class_grouping_applied: bool,
    class_grouping_reason: str,
) -> dict[str, object]:
    start = perf_counter()
    best_parameters, best_validation_f1 = _select_best_parameters(algorithm, parameter_grid, X_train, y_train, X_validation, y_validation, n_classes, seed)
    X_final = np.vstack([X_train, X_validation])
    y_final = np.concatenate([y_train, y_validation])
    model = create_classifier(algorithm, best_parameters, seed + 9999)
    model.fit(X_final, y_final, n_classes)
    predictions = model.predict(X_test)
    metrics = compute_classification_metrics(y_test, predictions, n_classes)

    return {
        "dataset": dataset.key,
        "dataset_title": dataset.title,
        "algorithm": algorithm,
        "run": run,
        "seed": seed,
        "best_validation_f1": float(best_validation_f1),
        "best_parameters": json.dumps(best_parameters, ensure_ascii=False),
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "elapsed_seconds": float(perf_counter() - start),
        "confusion_matrix": metrics.confusion_matrix.tolist(),
        "class_names": list(class_names),
        "class_grouping_applied": class_grouping_applied,
        "class_grouping_reason": class_grouping_reason,
    }


def _select_best_parameters(
    algorithm: str,
    parameter_grid: list[dict[str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[dict[str, object], float]:
    best_parameters = None
    best_validation_f1 = -1.0
    for configuration_index, parameters in enumerate(parameter_grid):
        model = create_classifier(algorithm, parameters, seed + configuration_index * 101)
        model.fit(X_train, y_train, n_classes)
        validation_predictions = model.predict(X_validation)
        validation_metrics = compute_classification_metrics(y_validation, validation_predictions, n_classes)
        if validation_metrics.f1_macro > best_validation_f1:
            best_validation_f1 = validation_metrics.f1_macro
            best_parameters = dict(parameters)
    if best_parameters is None:
        raise RuntimeError("No parameter configuration was evaluated.")
    return best_parameters, float(best_validation_f1)
