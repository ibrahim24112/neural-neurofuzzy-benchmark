from __future__ import annotations


def build_parameter_grids() -> dict[str, list[dict[str, object]]]:
    return {
        "Shallow_MLP": [
            {"hidden_layers": (12,), "activation": "tanh", "learning_rate": 0.010, "epochs": 150, "batch_size": 32, "l2_penalty": 1e-4},
            {"hidden_layers": (24,), "activation": "tanh", "learning_rate": 0.008, "epochs": 150, "batch_size": 64, "l2_penalty": 1e-4},
            {"hidden_layers": (32,), "activation": "relu", "learning_rate": 0.006, "epochs": 150, "batch_size": 64, "l2_penalty": 1e-4},
        ],
        "Deep_MLP": [
            {"hidden_layers": (24, 12), "activation": "tanh", "learning_rate": 0.008, "epochs": 170, "batch_size": 32, "l2_penalty": 1e-4},
            {"hidden_layers": (32, 16), "activation": "relu", "learning_rate": 0.006, "epochs": 170, "batch_size": 64, "l2_penalty": 1e-4},
            {"hidden_layers": (48, 24), "activation": "relu", "learning_rate": 0.004, "epochs": 170, "batch_size": 64, "l2_penalty": 5e-4},
        ],
        "Gaussian_Fuzzy_Rules": [
            {"rules_per_class": 1, "sigma_scale": 0.90},
            {"rules_per_class": 2, "sigma_scale": 1.10},
            {"rules_per_class": 3, "sigma_scale": 1.30},
        ],
        "RBF_TSK_NeuroFuzzy": [
            {"n_rules": 6, "sigma_scale": 0.90, "ridge_penalty": 1e-3},
            {"n_rules": 10, "sigma_scale": 1.10, "ridge_penalty": 1e-2},
            {"n_rules": 14, "sigma_scale": 1.30, "ridge_penalty": 1e-2},
        ],
    }
