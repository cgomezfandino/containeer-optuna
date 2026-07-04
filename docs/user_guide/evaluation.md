# Evaluation

## Metrics

- **Regression**: `regression_metrics(y_true, y_pred)` → `{r2, mse, rmse, mae}`.
  Optimize **R²** by default (maximize).
- **Clustering**: `clustering_metrics(X, labels)` →
  `{silhouette, calinski_harabasz, davies_bouldin, n_clusters}`.
  Optimize **Silhouette** (maximize); CH/DB stored as user attrs.

## Model cards

Every model ships with a structured card describing when to use it, pros,
cons, assumptions, complexity, and key hyperparameters. See [Model Cards](../model_cards/index.md).

```python
from containeer_optuna import get_model_card, all_model_cards
get_model_card("dbscan").cons
```
