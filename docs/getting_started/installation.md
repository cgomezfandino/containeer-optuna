# Installation

## Requirements

- Python ≥ 3.9 (3.10+ recommended)
- CPU is sufficient for M0; GPU support arrives in M6 (Deep Learning).

## From source (editable, recommended for development)

```bash
git clone https://github.com/cgomezfandino/containeer-optuna.git
cd containeer-optuna
pip install -e ".[dev]"
```

This installs runtime dependencies plus dev tooling (pytest, ruff, mypy,
pre-commit, mkdocs).

## Notebook environment

For the tutorial notebooks (Jupyter):

```bash
pip install -e ".[dev,notebooks]"
```

## Kaggle datasets

Some datasets (customer personality, credit card) require Kaggle credentials:

```bash
pip install kagglehub
# Place your Kaggle API token at ~/.kaggle/kaggle.json
```

## Verify the install

```bash
containeer-optuna --help
python -c "import containeer_optuna; print(containeer_optuna.__version__)"
```
