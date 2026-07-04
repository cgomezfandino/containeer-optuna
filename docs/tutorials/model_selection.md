# Model selection tutorial

Optuna can search over model families themselves, not just hyperparameters. The
original `more_advanced_workflow.ipynb` picks among Ridge / Lasso / OLS while
jointly tuning `alpha`/`tol`.

## Concept

Model type is a categorical hyperparameter:

```python
trial.suggest_categorical("model_type", ["ridge", "lasso", "ols"])
```

Conditional hyperparameters (`alpha`, `tol`) are only suggested when the chosen
model is Ridge or Lasso.

## In the framework

The framework's `get_model` + `suggest_params` dispatcher handles conditional
spaces generically. To run model selection across families, register multiple
models and let Optuna choose — see M1 (planned) for a one-line model-selection
study type.

For now, single-model studies are fully supported (one YAML per model).
