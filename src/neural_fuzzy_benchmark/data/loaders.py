from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neural_fuzzy_benchmark.data.catalog import DatasetSpec, get_dataset_specs


@dataclass(frozen=True)
class LoadedDataset:
    key: str
    title: str
    features: np.ndarray
    target: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    spec: DatasetSpec
    frame: pd.DataFrame


class DatasetLoader:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def load_all(self) -> dict[str, LoadedDataset]:
        return {spec.key: self.load(spec) for spec in get_dataset_specs()}

    def load(self, spec: DatasetSpec) -> LoadedDataset:
        if spec.key == "iris":
            return self._load_iris(spec)
        if spec.key == "wine":
            return self._load_wine(spec)
        if spec.key == "breast_cancer":
            return self._load_wdbc(spec)
        if spec.key == "seeds":
            return self._load_seeds(spec)
        raise ValueError(f"Unsupported dataset: {spec.key}")

    def _load_iris(self, spec: DatasetSpec) -> LoadedDataset:
        feature_names = tuple(spec.feature_descriptions)
        frame = pd.read_csv(self.data_dir / spec.raw_file, header=None, names=[*feature_names, spec.target_name])
        frame = frame.dropna().reset_index(drop=True)
        return self._build_dataset(spec, frame, feature_names)

    def _load_wine(self, spec: DatasetSpec) -> LoadedDataset:
        feature_names = tuple(spec.feature_descriptions)
        frame = pd.read_csv(self.data_dir / spec.raw_file, header=None, names=[spec.target_name, *feature_names])
        frame[spec.target_name] = frame[spec.target_name].map(lambda value: f"cultivar_{int(value)}")
        return self._build_dataset(spec, frame, feature_names)

    def _load_wdbc(self, spec: DatasetSpec) -> LoadedDataset:
        feature_names = tuple(spec.feature_descriptions)
        frame = pd.read_csv(self.data_dir / spec.raw_file, header=None, names=["id", spec.target_name, *feature_names])
        frame = frame.drop(columns=["id"])
        frame[spec.target_name] = frame[spec.target_name].map({"B": "benign", "M": "malignant"})
        return self._build_dataset(spec, frame, feature_names)

    def _load_seeds(self, spec: DatasetSpec) -> LoadedDataset:
        feature_names = tuple(spec.feature_descriptions)
        frame = pd.read_csv(self.data_dir / spec.raw_file, header=None, sep=r"\s+", names=[*feature_names, spec.target_name])
        frame[spec.target_name] = frame[spec.target_name].map(lambda value: f"wheat_{int(value)}")
        return self._build_dataset(spec, frame, feature_names)

    @staticmethod
    def _build_dataset(spec: DatasetSpec, frame: pd.DataFrame, feature_names: tuple[str, ...]) -> LoadedDataset:
        class_names = tuple(sorted(frame[spec.target_name].astype(str).unique()))
        class_to_int = {label: index for index, label in enumerate(class_names)}
        features = frame.loc[:, list(feature_names)].astype(float).to_numpy()
        target = frame[spec.target_name].astype(str).map(class_to_int).to_numpy(dtype=int)
        return LoadedDataset(
            key=spec.key,
            title=spec.title,
            features=features,
            target=target,
            feature_names=feature_names,
            class_names=class_names,
            spec=spec,
            frame=frame,
        )
