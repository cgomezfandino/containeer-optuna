# Evaluation

## Regression metrics (pluggable — M1)

Set the `metric` field on an experiment to choose the optimization objective.
The direction is derived automatically:

| Metric | Direction | Scorer | Notes |
|--------|-----------|--------|-------|
| `r2` (default) | maximize | `r2` | Coefficient of determination ∈ (-∞, 1]. |
| `mse` | minimize | `neg_mean_squared_error` | Mean squared error. |
| `rmse` | minimize | `neg_root_mean_squared_error` | Root mean squared error. |
| `mae` | minimize | `neg_mean_absolute_error` | Robust to outliers. |

```yaml
metric: mse   # → optimization.direction becomes "minimize" automatically
```

```python
from containeer_optuna import get_regression_scorer
scorer, direction = get_regression_scorer("mae")  # → (scorer, "minimize")
```

## Classification metrics (M4)

All classification metrics are **maximize**. Binary metrics (`f1`, `roc_auc`)
require 0/1 targets; use `f1_weighted` and `roc_auc_ovr` for multiclass:

| Metric | Direction | Scope | Notes |
|--------|-----------|-------|-------|
| `accuracy` (default) | maximize | binary + multiclass | Fraction correct. |
| `f1_weighted` | maximize | multiclass | Weighted F1 across classes. |
| `roc_auc_ovr` | maximize | multiclass | One-vs-rest AUC. |
| `f1` | maximize | binary only | Requires 0/1 targets. |
| `roc_auc` | maximize | binary only | Requires `probability=True` for SVC. |

```yaml
task: classification
metric: f1_weighted   # → direction: maximize, scorer: f1_weighted
```

```python
from containeer_optuna import get_classification_scorer
scorer, direction = get_classification_scorer("accuracy")  # → (scorer, "maximize")
```

## Clustering metrics

`clustering_metrics(X, labels)` returns Silhouette (maximize, primary),
Calinski-Harabasz (maximize, user_attr), and Davies-Bouldin (minimize,
user_attr).

## Visualization (M3)

- `plot_embedding_2d(X_2d, labels)` — 2D scatter colored by cluster labels.
- `plot_scree(explained_variance_ratio)` — explained-variance chart.
- `runner.plot_best_embedding()` — rebuilds the best trial's pipeline and
  projects `X` to 2D (works with ALL reducers including t-SNE).

## Model cards

Every model ships with a structured card (when to use, pros, cons, assumptions,
complexity, key hyperparameters). See [Model Cards](../model_cards/index.md).

```python
from containeer_optuna import get_model_card
get_model_card("random_forest").cons
```
