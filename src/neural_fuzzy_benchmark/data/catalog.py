from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    raw_file: str
    task_type: str
    target_name: str
    source: str
    url: str
    description: str
    feature_descriptions: dict[str, str]
    target_description: str


IRIS_FEATURES = {
    "sepal_length_cm": "Sepal length measured in centimeters.",
    "sepal_width_cm": "Sepal width measured in centimeters.",
    "petal_length_cm": "Petal length measured in centimeters.",
    "petal_width_cm": "Petal width measured in centimeters.",
}

WINE_FEATURES = {
    "alcohol": "Alcohol content.",
    "malic_acid": "Malic acid concentration.",
    "ash": "Ash content.",
    "alcalinity_of_ash": "Alcalinity of ash.",
    "magnesium": "Magnesium concentration.",
    "total_phenols": "Total phenols.",
    "flavanoids": "Flavanoid concentration.",
    "nonflavanoid_phenols": "Nonflavanoid phenols.",
    "proanthocyanins": "Proanthocyanins.",
    "color_intensity": "Color intensity.",
    "hue": "Hue.",
    "od280_od315": "OD280/OD315 of diluted wines.",
    "proline": "Proline concentration.",
}

WDBC_MEASURES = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
)

WDBC_FEATURES = {
    f"{stat}_{measure}": f"{stat.title()} value of {measure.replace('_', ' ')} computed from cell nucleus measurements."
    for stat in ("mean", "se", "worst")
    for measure in WDBC_MEASURES
}

SEEDS_FEATURES = {
    "area": "Kernel area.",
    "perimeter": "Kernel perimeter.",
    "compactness": "Compactness coefficient.",
    "kernel_length": "Length of kernel.",
    "kernel_width": "Width of kernel.",
    "asymmetry_coefficient": "Kernel asymmetry coefficient.",
    "kernel_groove_length": "Length of the kernel groove.",
}

DATASET_SPECS = (
    DatasetSpec(
        key="iris",
        title="Iris Plants",
        raw_file="iris.data",
        task_type="Multiclass classification",
        target_name="species",
        source="UCI Machine Learning Repository",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        description="Classify Iris flower species using continuous sepal and petal measurements.",
        feature_descriptions=IRIS_FEATURES,
        target_description="Iris species: setosa, versicolor or virginica.",
    ),
    DatasetSpec(
        key="wine",
        title="Wine Recognition",
        raw_file="wine.data",
        task_type="Multiclass classification",
        target_name="cultivar",
        source="UCI Machine Learning Repository",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        description="Identify wine cultivar from continuous chemical analysis measurements.",
        feature_descriptions=WINE_FEATURES,
        target_description="Wine cultivar group. The selected source contains three cultivar classes.",
    ),
    DatasetSpec(
        key="breast_cancer",
        title="Breast Cancer Wisconsin Diagnostic",
        raw_file="wdbc.data",
        task_type="Binary classification",
        target_name="diagnosis",
        source="UCI Machine Learning Repository",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        description="Classify breast tumors as benign or malignant using continuous cell nucleus measurements.",
        feature_descriptions=WDBC_FEATURES,
        target_description="Diagnosis: benign or malignant.",
    ),
    DatasetSpec(
        key="seeds",
        title="Seeds",
        raw_file="seeds_dataset.txt",
        task_type="Multiclass classification",
        target_name="wheat_type",
        source="UCI Machine Learning Repository",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        description="Classify wheat varieties from continuous geometric measurements extracted from kernel images.",
        feature_descriptions=SEEDS_FEATURES,
        target_description="Wheat variety class.",
    ),
)


def get_dataset_specs() -> tuple[DatasetSpec, ...]:
    return DATASET_SPECS
