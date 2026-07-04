# Classification tutorial

M4 adds full classification support: 4 classifiers, classification metrics,
StratifiedKFold by default, and bundled classification datasets.

## Classifiers

| Model | Milestone | Key hyperparameters |
|-------|-----------|---------------------|
| `logistic_regression` | M4 | `C`, `penalty`, `solver` |
| `knn` | M4 | `n_neighbors`, `weights`, `p` |
| `svc` | M4 | `C`, `kernel`, `gamma` |
| `decision_tree_classifier` | M4 | `max_depth`, `min_samples_split`, `criterion` |

## Metrics

All classification metrics are **maximize**:

| Metric | Scope | Notes |
|--------|-------|-------|
| `accuracy` (default) | binary + multiclass | Fraction correct. |
| `f1_weighted` | multiclass | Weighted F1 across classes. |
| `roc_auc_ovr` | multiclass | One-vs-rest AUC. |
| `f1` | binary only | Requires 0/1 targets. |
| `roc_auc` | binary only | Requires `probability=True` for SVC. |

## Datasets

Bundled (no download):

| Dataset | Classes | Samples | Features |
|---------|---------|---------|----------|
| `breast_cancer` | 2 (binary) | 569 | 30 |
| `wine` | 3 | 178 | 13 |
| `iris_classification` | 3 | 150 | 4 |

## Run LogisticRegression on breast_cancer

```bash
containeer-optuna run config/experiments/breast_cancer_classification.yaml --n-trials 30
```

## From Python

```python
from containeer_optuna import load_config, OptunaRunner

cfg = load_config("config/experiments/breast_cancer_classification.yaml")
study = OptunaRunner(cfg).run(n_trials=30)
print("best accuracy:", study.best_value)
```

## Model selection across classifier families

```bash
containeer-optuna run config/experiments/classification_model_selection.yaml --n-trials 60
```

This searches across LogisticRegression, KNN, SVC, and DecisionTreeClassifier
in a single Optuna study. The `model_type` parameter reveals which family won.

## Auto-derived defaults

When `task: classification`, the framework:
- Defaults `cv.strategy` to `stratified_kfold` (preserves class proportions).
- Derives `optimization.direction` from the metric (all classification
  metrics → maximize).

```yaml
task: classification
metric: f1_weighted   # → direction: maximize, scorer: f1_weighted
```
