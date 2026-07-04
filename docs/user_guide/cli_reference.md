# CLI Reference

The `containeer-optuna` CLI is built with [Typer](https://typer.tiangolo.com/).

## `run` — Execute an experiment

```bash
containeer-optuna run <config.yaml> [--n-trials N] [--no-progress]
```

Loads the YAML, runs the Optuna study, prints the best trial summary.

## `list-models`

```bash
containeer-optuna list-models [--type clustering|regression|classification|reducer|scaler]
```

## `list-datasets`

Lists datasets registered in `config/datasets.yaml`.

## `list-experiments`

Lists experiment YAMLs in `config/experiments/`.

## `describe <model>`

Prints the model card (summary, when to use, pros, cons, assumptions,
complexity, key hyperparameters).

## `init <name>`

Scaffolds a new experiment YAML.

## `dashboard`

Prints the `optuna-dashboard` command for the given storage URL.

## `stats` — Statistical analysis (M5)

A subcommand group for quick exploratory analysis from the terminal:

```bash
containeer-optuna stats describe <dataset>
containeer-optuna stats ttest <dataset> --group-by <col> --feature <col>
containeer-optuna stats anova <dataset> --group-by <col> --feature <col>
containeer-optuna stats correlation <dataset> --method pearson --threshold 0.3
containeer-optuna stats normality <dataset> --feature <col>
```

See [Statistics](statistics.md) for the full Python API.
