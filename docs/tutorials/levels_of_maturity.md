# Levels of maturity in hyperparameter tuning

Adapted from `notebooks/optuna/hyperparam_tuning_levels_of_maturity.ipynb`,
which walks through four progressively sophisticated approaches.

## Level 1 — What even are hyperparameters?

A single fit with fixed hyperparameters (e.g. `Ridge(alpha=2.0)`).
Establishes the distinction between *hyperparameters* (set before training,
like `alpha`) and *learned parameters* (fit during training, like coefficients).

## Level 2 — Manual tuning

Hand-written loops over a handful of hardcoded values. Simple but unscalable:
combinatorial explosion kills it past 2–3 hyperparameters.

## Level 3 — Grid / Random search

`GridSearchCV` (exhaustive cartesian product) and `RandomizedSearchCV` (sample
from distributions). Better coverage, but still uninformed — every evaluation
is independent of previous results.

## Level 4 — Optuna (Bayesian optimization)

TPE (Tree-structured Parzen Estimator) builds a probabilistic model of the
objective and *uses it to pick the next trial*. Far more sample-efficient than
grid/random on expensive evaluations.

## Why this framework standardizes on Level 4

Every experiment in `containeer_optuna` is an Optuna study. The same YAML,
the same runner, the same storage/dashboard — whether you tune one
hyperparameter or twenty.
