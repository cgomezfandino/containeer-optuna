# Models

Models are registered in `config/models.yaml` with `default_params` and a
`param_space` (the Optuna search spec). See [Model Cards](../model_cards/index.md)
for pros/cons and when-to-use guidance.

## Registered models (M0)

| Name | Kind | Optuna-tunable params |
|------|------|----------------------|
| `ridge` | regression | `alpha`, `tol` |
| `lasso` | regression | `alpha`, `tol` |
| `ols` | regression | (none) |
| `kmeans` | clustering | `n_clusters`, `init`, `algorithm` |
| `dbscan` | clustering | `eps`, `min_samples` |
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
