# Models

Models are registered in `config/models.yaml` with `default_params` and a
`param_space` (the Optuna search spec). See [Model Cards](../model_cards/index.md)
for pros/cons and when-to-use guidance.

## Registered models

### Regression (M0 + M1)

| Name | Milestone | Optuna-tunable params |
|------|-----------|----------------------|
| `ridge` | M0 | `alpha`, `tol` |
| `lasso` | M0 | `alpha`, `tol` |
| `ols` | M0 | (none) |
| `elasticnet` | M1 | `alpha`, `l1_ratio` |
| `decision_tree` | M1 | `max_depth`, `min_samples_split` |
| `random_forest` | M1 | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| `gradient_boosting` | M1 | `learning_rate`, `n_estimators`, `max_depth`, `subsample` |
| `svr` | M1 | `C`, `epsilon`, `kernel` |

### Clustering (M0)

| Name | Kind | Optuna-tunable params |
|------|------|----------------------|
| `kmeans` | clustering | `n_clusters`, `init`, `algorithm` |
| `dbscan` | clustering | `eps`, `min_samples`` |
| `gmm` | clustering | `n_components`, `covariance_type` |
| `pca` | reducer | `n_components` |
| `umap` | reducer | `n_neighbors`, `min_dist`, `n_components` |
| `standard_scaler` | scaler | (none) |
| `minmax_scaler` | scaler | (none) |

## Building a model in Python

```python
from containeer_optuna import get_model

# Default params
est = get_model("ridge")

# Sampled from param_space (inside an Optuna trial)
est = get_model("kmeans", trial=trial, namespace="kmeans")
```

## Adding a new model

1. Add an entry to `config/models.yaml` with `type`, `default_params`, `param_space`.
2. Add the sklearn class to `src/containeer_optuna/models/classes.py`.
3. Add a model card in `src/containeer_optuna/evaluation/model_cards.py`.
