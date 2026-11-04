from __future__ import annotations

import pandas as pd

from neural_fuzzy_benchmark.data.catalog import get_dataset_specs
from neural_fuzzy_benchmark.data.loaders import LoadedDataset


def build_dataset_metadata_table(datasets: dict[str, LoadedDataset]) -> pd.DataFrame:
    rows = []
    for dataset in datasets.values():
        counts = pd.Series(dataset.target).value_counts().sort_index()
        distribution = {dataset.class_names[int(index)]: int(value) for index, value in counts.items()}
        rows.append(
            {
                "dataset": dataset.key,
                "title": dataset.title,
                "source": dataset.spec.source,
                "url": dataset.spec.url,
                "task_type": dataset.spec.task_type,
                "target": dataset.spec.target_name,
                "samples": int(dataset.features.shape[0]),
                "features": int(dataset.features.shape[1]),
                "classes": int(len(dataset.class_names)),
                "class_distribution": distribution,
                "description": dataset.spec.description,
            }
        )
    return pd.DataFrame(rows)


def build_data_dictionary_table() -> pd.DataFrame:
    rows = []
    for spec in get_dataset_specs():
        rows.append(
            {
                "dataset": spec.key,
                "variable": spec.target_name,
                "role": "target",
                "description": spec.target_description,
                "scale": "categorical",
            }
        )
        for feature_name, description in spec.feature_descriptions.items():
            rows.append(
                {
                    "dataset": spec.key,
                    "variable": feature_name,
                    "role": "feature",
                    "description": description,
                    "scale": "continuous numeric",
                }
            )
    return pd.DataFrame(rows)


def build_class_distribution_table(datasets: dict[str, LoadedDataset]) -> pd.DataFrame:
    rows = []
    for dataset in datasets.values():
        counts = pd.Series(dataset.target).value_counts().sort_index()
        total = counts.sum()
        for class_index, count in counts.items():
            rows.append(
                {
                    "dataset": dataset.key,
                    "class": dataset.class_names[int(class_index)],
                    "samples": int(count),
                    "share": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def build_feature_summary_table(datasets: dict[str, LoadedDataset]) -> pd.DataFrame:
    rows = []
    for dataset in datasets.values():
        frame = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        description = frame.describe().T
        for feature, values in description.iterrows():
            rows.append(
                {
                    "dataset": dataset.key,
                    "feature": feature,
                    "mean": float(values["mean"]),
                    "std": float(values["std"]),
                    "min": float(values["min"]),
                    "q25": float(values["25%"]),
                    "median": float(values["50%"]),
                    "q75": float(values["75%"]),
                    "max": float(values["max"]),
                }
            )
    return pd.DataFrame(rows)
