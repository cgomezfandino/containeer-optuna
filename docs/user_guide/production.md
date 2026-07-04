# Productionization

## Model serialization

Set `save_model: true` in your experiment YAML to persist the best model after
optimization:

```yaml
save_model: true
```

The model is saved to `results/<experiment_name>_best_model.joblib` using
joblib. It can be loaded back:

```python
import joblib
model = joblib.load("results/my_experiment_best_model.joblib")
predictions = model.predict(X_new)
```

You can also reconstruct and fit the best pipeline manually:

```python
runner = OptunaRunner(cfg)
runner.run(n_trials=30)
pipeline = runner.fit_best_pipeline()  # reconstruct from best_params + fit on full data
```

**Note**: `fit_best_pipeline` works for sklearn models only (regression,
clustering, classification). DL/NLP models use custom training loops and
raise `NotImplementedError`.

## Predictions

Set `save_predictions: true` to persist predictions:

```yaml
save_predictions: true
```

Saved to `results/<experiment_name>_predictions.npy`.

## MLflow tracking

Set `tracking: mlflow` in the optimization block:

```yaml
optimization:
  tracking: mlflow
```

Requires `pip install containeer-optuna[mlflow]`. Logs:
- All best-trial hyperparameters.
- `best_value` metric.
- User attrs (e.g. `mean_calinski_harabasz`, `mean_r2`).

## Python module

```bash
python -m containeer_optuna --help
python -m containeer_optuna run config/experiments/diabetes_regression.yaml
```

## PyPI install

```bash
pip install containeer-optuna            # core ML + statistics
pip install containeer-optuna[dl]        # + PyTorch MLP/CNN/RNN
pip install containeer-optuna[nlp]       # + HuggingFace transformers
pip install containeer-optuna[mlflow]    # + MLflow tracking
```
