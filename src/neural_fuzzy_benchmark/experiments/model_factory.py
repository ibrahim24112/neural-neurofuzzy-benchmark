from __future__ import annotations

from neural_fuzzy_benchmark.models.gaussian_fuzzy_rules import GaussianFuzzyRulesClassifier
from neural_fuzzy_benchmark.models.mlp import MLPClassifier
from neural_fuzzy_benchmark.models.rbf_tsk_neurofuzzy import RbfTskNeuroFuzzyClassifier


def create_classifier(algorithm: str, parameters: dict[str, object], seed: int):
    if algorithm in {"Shallow_MLP", "Deep_MLP"}:
        return MLPClassifier(
            hidden_layers=tuple(parameters["hidden_layers"]),
            activation=str(parameters["activation"]),
            learning_rate=float(parameters["learning_rate"]),
            epochs=int(parameters["epochs"]),
            batch_size=int(parameters["batch_size"]),
            l2_penalty=float(parameters["l2_penalty"]),
            seed=seed,
        )
    if algorithm == "Gaussian_Fuzzy_Rules":
        return GaussianFuzzyRulesClassifier(
            rules_per_class=int(parameters["rules_per_class"]),
            sigma_scale=float(parameters["sigma_scale"]),
            seed=seed,
        )
    if algorithm == "RBF_TSK_NeuroFuzzy":
        return RbfTskNeuroFuzzyClassifier(
            n_rules=int(parameters["n_rules"]),
            sigma_scale=float(parameters["sigma_scale"]),
            ridge_penalty=float(parameters["ridge_penalty"]),
            seed=seed,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")
