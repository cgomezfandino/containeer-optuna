# Changelog

## [Unreleased] — M6 — Deep Learning Foundation

### Added
- **PyTorch MLP** for tabular regression (`mlp_regressor`) and classification
  (`mlp_classifier`). Requires `pip install containeer-optuna[dl]`.
- **`make_dl_objective`**: a custom DL objective that bypasses `cross_validate`
  and runs a manual PyTorch training loop with **epoch pruning** (`trial.report`
  + `trial.should_prune`). Mirrors the clustering objective's manual-fold pattern.
- **`models/dl/` subpackage**: `build_mlp_module` (flexible `nn.Module`) +
  `get_loss_fn` (MSELoss / CrossEntropyLoss).
- **`[dl]` optional extra** in `pyproject.toml` (`torch>=2.0.0`).
- **Lazy torch import**: package imports fine without torch; DL models raise
  a clear ImportError when accessed.
- **Experiments**: `diabetes_mlp_regression.yaml`, `breast_cancer_mlp_classification.yaml`
  (both with `pruner: median`).
- **7 new tests** (174 total): MLP forward pass, DL regression e2e, DL
  classification e2e, epoch pruning fires, model cards. All guarded by
  `pytest.importorskip("torch")`.

### Design
- DL models bypass `cross_validate` (incompatible with PyTorch training loops).
- The DL objective iterates CV folds manually and reports val_loss per epoch
  for pruning — the main advantage of Optuna for DL.
- `hidden_layer_sizes` expressed as categorical choices of lists
  (`[[64], [128, 64], [256, 128, 64]]`).

## [0.6.0] — M5 — Statistics

scipy.stats-based hypothesis tests, normality, correlation, CLI stats group.
See PR #6.

## [0.5.0] — M4 — Classification

4 classifiers, classification metrics, StratifiedKFold, datasets. See PR #5.

## [0.4.0] — M3 — Dimensionality Reduction

3 new reducers, visualization, plot_best_embedding. See PR #4.

## [0.3.0] — M2 — Clustering Maturity

5 clustering models, model-selection-as-categorical, DBSCAN fix. See PR #3.

## [0.2.0] — M1 — Regression Maturity

5 regression models, pluggable metrics, diabetes dataset, feature-set
selection. See PR #2.

## [0.1.0] — M0 — Foundation

See PR #1.
