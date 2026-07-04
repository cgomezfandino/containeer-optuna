# Config Reference

See [Configuration (Getting Started)](../getting_started/configuration.md) for
the YAML schema and `containeer_optuna.config` in the API reference for the
full pydantic model definitions.

## Key models

- `ExperimentConfig` — top-level experiment.
- `CVConfig` — cross-validation (`shuffle_split`, `kfold`, `stratified_kfold`).
- `OptimizationConfig` — Optuna study (`sampler`, `pruner`, `storage`, ...).
- `DatasetConfig` — dataset entry in `datasets.yaml`.
- `ModelConfig` — model entry in `models.yaml`.
- `Settings` — env-driven global settings (`CO_` prefix).
