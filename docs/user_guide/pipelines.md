# Pipelines

Pipelines chain optional preprocessing (scaler, reducer) with a final model.
They are built from registry names so YAML configs stay concise.

## Assembly order

```
[scaler?] -> [reducer?] -> model
```

The order is fixed: scaling happens before reduction, reduction before the
model. This avoids leakage (scaler is fitted inside CV, not globally).

## Building in Python

```python
from containeer_optuna import get_pipeline

# Full clustering pipeline
p = get_pipeline("kmeans", scaler="standard_scaler", reducer="pca")
labels = p.fit_predict(X)

# Inside an Optuna trial
p = get_pipeline("kmeans", scaler="standard_scaler", trial=trial)
```

## Namespacing

When both the reducer and the model are tunable, each gets its own namespace
(`pca_n_components`, `kmeans_n_clusters`) to keep the Optuna search space flat
and unambiguous.
