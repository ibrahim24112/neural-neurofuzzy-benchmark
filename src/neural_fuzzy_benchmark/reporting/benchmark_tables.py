from __future__ import annotations

import json

import pandas as pd


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
    "elapsed_seconds": "Elapsed seconds",
}


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    std = 0.0 if pd.isna(std) else std
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_benchmark_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "algorithm": row["algorithm"],
                "accuracy": format_mean_std(row["accuracy_mean"], row["accuracy_std"]),
                "macro_precision": format_mean_std(row["precision_macro_mean"], row["precision_macro_std"]),
                "macro_recall": format_mean_std(row["recall_macro_mean"], row["recall_macro_std"]),
                "macro_f1": format_mean_std(row["f1_macro_mean"], row["f1_macro_std"]),
                "elapsed_seconds": format_mean_std(row["elapsed_seconds_mean"], row["elapsed_seconds_std"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "macro_f1"], ascending=[True, False]).reset_index(drop=True)


def build_metric_matrix(summary: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    table = summary.copy()
    table[METRIC_LABELS.get(metric, metric)] = [format_mean_std(mean, std) for mean, std in zip(table[mean_column], table[std_column])]
    return table.pivot(index="dataset", columns="algorithm", values=METRIC_LABELS.get(metric, metric)).reset_index()


def rank_algorithms_by_dataset(summary: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    mean_column = f"{metric}_mean"
    ranking = summary[["dataset", "algorithm", mean_column]].copy()
    ranking["rank"] = ranking.groupby("dataset")[mean_column].rank(ascending=False, method="min").astype(int)
    return ranking.sort_values(["dataset", "rank", "algorithm"]).reset_index(drop=True)


def build_best_parameter_frequency(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, algorithm), group in results.groupby(["dataset", "algorithm"]):
        counts = group["best_parameters"].value_counts()
        for parameters, count in counts.items():
            parsed = json.loads(parameters) if isinstance(parameters, str) else parameters
            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "best_parameters": parsed,
                    "frequency": int(count),
                    "share": float(count / len(group)),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "algorithm", "frequency"], ascending=[True, True, False]).reset_index(drop=True)


def build_model_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "algorithm": "Shallow_MLP",
                "family": "Artificial neural network",
                "main_idea": "One hidden layer learns nonlinear decision boundaries by optimizing weights.",
                "key_parameters": "hidden_layers, activation, learning_rate, epochs, batch_size, l2_penalty",
                "strength": "Good predictive capacity with moderate complexity.",
                "limitation": "Less interpretable than fuzzy rules.",
            },
            {
                "algorithm": "Deep_MLP",
                "family": "Artificial neural network",
                "main_idea": "Two hidden layers increase representational capacity.",
                "key_parameters": "hidden_layers, activation, learning_rate, epochs, batch_size, l2_penalty",
                "strength": "More flexible for nonlinear structure.",
                "limitation": "Higher sensitivity to hyperparameters and initialization.",
            },
            {
                "algorithm": "Gaussian_Fuzzy_Rules",
                "family": "Fuzzy system",
                "main_idea": "Class prototypes generate Gaussian membership rules; class scores are rule activations.",
                "key_parameters": "rules_per_class, sigma_scale",
                "strength": "Interpretable rules based on centers and spreads.",
                "limitation": "Limited flexibility when class boundaries are highly complex.",
            },
            {
                "algorithm": "RBF_TSK_NeuroFuzzy",
                "family": "Neuro-fuzzy system",
                "main_idea": "RBF antecedents activate TSK consequents fitted by ridge regularization.",
                "key_parameters": "n_rules, sigma_scale, ridge_penalty",
                "strength": "Balances fuzzy activations with trainable consequents.",
                "limitation": "Less transparent than direct fuzzy voting rules.",
            },
        ]
    )
