# Changelog

## [Unreleased] — M4 — Classification

### Added
- **4 classifiers** with industry-aligned hyperparameter ranges + full model
  cards: logistic_regression, knn, svc, decision_tree_classifier.
- **Classification metrics**: accuracy, f1, f1_weighted, roc_auc, roc_auc_ovr
  (all maximize). New `CLASSIFICATION_SCORERS` + `get_classification_scorer`.
- **`make_classification_objective`**: mirrors the regression objective with
  classification scorers; supports feature_sets + StratifiedKFold.
- **Classification model-selection**: the `models: [...]` field now works for
  classification (searches across LogReg/KNN/SVC/DTClassifier in one study).
- **3 bundled classification datasets**: breast_cancer (binary, 569×30), wine
  (3-class, 178×13), iris_classification (3-class, 150×4). No download.
- **Experiments**: breast_cancer_classification.yaml,
  classification_model_selection.yaml.
- **23 new tests** (147 total): classifier instantiation + sampled params,
  classification scorers, metric→direction, stratified_kfold default,
  classification e2e (parametrized across the 4 classifiers),
  classification model-selection e2e.

### Changed
- `ExperimentConfig.metric` widened to accept both regression and
  classification metrics.
- `ExperimentConfig` validator now defaults `cv.strategy` to
  `stratified_kfold` for classification (mirrors the clustering→kfold default).
- `METRIC_DIRECTION` extended with the 5 classification metrics.
- The model-selection objective now handles classification (cross_validate
  branch shared with regression).

## [0.4.0] — M3 — Dimensionality Reduction

3 new reducers, visualization, plot_best_embedding. See PR #4.

## [0.3.0] — M2 — Clustering Maturity

5 clustering models, model-selection-as-categorical, DBSCAN fix. See PR #3.

## [0.2.0] — M1 — Regression Maturity

5 regression models, pluggable metrics, diabetes dataset, feature-set
selection. See PR #2.

## [0.1.0] — M0 — Foundation

See PR #1.
