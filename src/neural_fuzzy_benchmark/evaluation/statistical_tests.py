from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def friedman_tests_by_dataset(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for dataset, dataset_results in results.groupby("dataset"):
        pivot = _metric_pivot(dataset_results, metric)
        statistic, p_value = friedmanchisquare(*[pivot[column].to_numpy() for column in pivot.columns])
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "n_algorithms": int(len(pivot.columns)),
                "n_blocks": int(len(pivot)),
            }
        )
    return pd.DataFrame(rows)


def global_friedman_test(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = _metric_pivot(results, metric)
    statistic, p_value = friedmanchisquare(*[pivot[column].to_numpy() for column in pivot.columns])
    return pd.DataFrame(
        [
            {
                "scope": "all_datasets",
                "metric": metric,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "n_algorithms": int(len(pivot.columns)),
                "n_blocks": int(len(pivot)),
            }
        ]
    )


def pairwise_wilcoxon_tests_by_dataset(results: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    frames = []
    for dataset, dataset_results in results.groupby("dataset"):
        table = _pairwise_wilcoxon(dataset_results, metric, alpha)
        table.insert(0, "dataset", dataset)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def global_pairwise_wilcoxon_tests(results: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    table = _pairwise_wilcoxon(results, metric, alpha)
    table.insert(0, "scope", "all_datasets")
    return table


def _pairwise_wilcoxon(results: pd.DataFrame, metric: str, alpha: float) -> pd.DataFrame:
    pivot = _metric_pivot(results, metric)
    rows = []
    for algorithm_a, algorithm_b in combinations(pivot.columns, 2):
        values_a = pivot[algorithm_a].to_numpy()
        values_b = pivot[algorithm_b].to_numpy()
        try:
            statistic, p_value = wilcoxon(values_a, values_b, zero_method="zsplit")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        rows.append(
            {
                "metric": metric,
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "mean_difference_a_minus_b": float(np.mean(values_a - values_b)),
                "statistic": float(statistic),
                "p_value_raw": float(p_value),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_value_bonferroni"] = np.minimum(table["p_value_raw"] * len(table), 1.0)
    table["significant_bonferroni"] = table["p_value_bonferroni"] < alpha
    table["p_value_holm"] = _holm_adjust(table["p_value_raw"].to_numpy())
    table["significant_holm"] = table["p_value_holm"] < alpha
    return table


def _metric_pivot(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = results.pivot_table(index=["dataset", "run"], columns="algorithm", values=metric, aggfunc="mean").dropna()
    if pivot.shape[1] < 2:
        raise ValueError("At least two algorithms are required for statistical comparison.")
    return pivot


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    previous = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        adjusted_value = min((total - rank) * p_values[original_index], 1.0)
        previous = max(previous, adjusted_value)
        adjusted[original_index] = previous
    return adjusted
