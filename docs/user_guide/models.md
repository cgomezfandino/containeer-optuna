# Models

Models are registered in `config/models.yaml` with `default_params` and a
`param_space` (the Optuna search spec). See [Model Cards](../model_cards/index.md)
for pros/cons and when-to-use guidance.

## Registered models (27 total)

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

### Classification (M4)

| Name | Milestone | Optuna-tunable params |
|------|-----------|----------------------|
| `logistic_regression` | M4 | `C`, `penalty`, `solver` |
| `knn` | M4 | `n_neighbors`, `weights`, `p` |
| `svc` | M4 | `C`, `kernel`, `gamma` |
| `decision_tree_classifier` | M4 | `max_depth`, `min_samples_split`, `criterion` |

### Clustering (M0 + M2)

| Name | Milestone | Optuna-tunable params |
|------|-----------|----------------------|
| `kmeans` | M0 | `n_clusters`, `init`, `algorithm` |
| `dbscan` | M0 | `eps`, `min_samples` |
| `gmm` | M0 | `n_components`, `covariance_type` |
| `hdbscan` | M2 | `min_cluster_size`, `min_samples`, `cluster_selection_method` |
| `agglomerative` | M2 | `n_clusters`, `linkage` |
| `spectral` | M2 | `n_clusters`, `affinity` |
| `birch` | M2 | `n_clusters`, `threshold` |
| `optics` | M2 | `min_samples`, `xi` |

### Reducers (M0 + M3)

| Name | Milestone | Optuna-tunable params |
|------|-----------|----------------------|
| `pca` | M0 | `n_components` |
| `umap` | M0 | `n_neighbors`, `min_dist`, `n_components` |
| `tsne` | M3 | `n_components`, `perplexity`, `learning_rate` |
| `truncated_svd` | M3 | `n_components` |
| `factor_analysis` | M3 | `n_components` |

### Scalers (M0)

| Name | Optuna-tunable params |
|------|----------------------|
| `standard_scaler` | (none) |
| `minmax_scaler` | (none) |

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
4. Add a test that it instantiates.
