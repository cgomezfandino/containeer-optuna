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

In Python:

```python
from containeer_optuna import get_regression_scorer
scorer, direction = get_regression_scorer("mae")  # → (scorer, "minimize")
```

## Clustering metrics

`clustering_metrics(X, labels)` returns Silhouette (maximize, primary),
Calinski-Harabasz (maximize, user_attr), and Davies-Bouldin (minimize,
user_attr).

## Model cards

Every model ships with a structured card (when to use, pros, cons, assumptions,
complexity, key hyperparameters). See [Model Cards](../model_cards/index.md).

```python
from containeer_optuna import get_model_card
get_model_card("random_forest").cons
```
