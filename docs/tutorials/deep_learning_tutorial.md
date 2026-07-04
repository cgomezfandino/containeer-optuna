# Deep learning tutorial (M6)

The framework supports PyTorch MLP models for tabular regression and
classification with **epoch pruning** — the main advantage of Optuna for DL.

## Prerequisites

```bash
pip install containeer-optuna[dl]
```

## MLP regression

```bash
containeer-optuna run config/experiments/diabetes_mlp_regression.yaml --n-trials 30
```

The experiment YAML uses `pruner: median` for epoch pruning:

```yaml
model: mlp_regressor
pruner: median
```

## How epoch pruning works

Each trial trains the MLP for `epochs` epochs. After each epoch, the objective
calls `trial.report(val_loss, epoch)` and checks `trial.should_prune()`. If the
validation loss is worse than the median of completed trials at that step,
Optuna prunes the trial early — saving compute on bad hyperparameter configs.

## MLP classification

```bash
containeer-optuna run config/experiments/breast_cancer_mlp_classification.yaml --n-trials 30
```

## When to use MLP vs sklearn models

MLPs shine when:
- Linear models underfit (non-linear feature interactions).
- Tree ensembles aren't ideal (very wide data, smooth target).
- You want calibrated probabilities with complex boundaries.

Use sklearn models (RandomForest, GradientBoosting) by default — they need
fewer trials and no GPU. Switch to MLP only when sklearn models plateau.
