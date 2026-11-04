# Neural and Neuro-Fuzzy Benchmark

An experimental benchmark comparing artificial neural networks, fuzzy-rule systems, and neuro-fuzzy models across four real tabular classification datasets: Iris, Wine, Breast Cancer Wisconsin Diagnostic, and Seeds.

## Models

- Shallow and deep multilayer perceptrons
- Gaussian fuzzy rules
- RBF antecedents with Takagi–Sugeno–Kang-style consequents

## Experimental design

The workflow documents the datasets, explores their feature and class distributions, and uses separate training, validation, and final-test partitions. Hyperparameters are selected on validation data; the held-out test split is evaluated only after selection. The reported protocol repeats experiments with independent seeds and summarizes results with descriptive statistics and corrected non-parametric comparisons.

## Repository layout

```text
data/raw/                         bundled source datasets
src/neural_fuzzy_benchmark/       reusable implementation
notebooks/neural_fuzzy_benchmark.ipynb
requirements.txt
README.md
```

Notebook outputs and machine-specific execution metadata were removed in this portfolio copy. The notebook retains its explanations and code cells; run it sequentially to recreate tables and figures.

## Setup and use

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
jupyter lab notebooks/neural_fuzzy_benchmark.ipynb
```

## Data and responsible use

The bundled datasets are documented in the notebook and originate from commonly used public machine-learning repositories. Verify each dataset's current citation and reuse terms before redistributing it or publishing the repository.

## Publication note

The project files were curated but not executed or tested during sorting. Confirm dependency versions, dataset provenance, and a clean end-to-end run before public release.