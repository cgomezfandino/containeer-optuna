# Changelog

## [0.6.0] — M5 — Statistics

### Added
- **`statistics/` subpackage** — hypothesis tests, normality tests,
  correlation, chi-square, and descriptive statistics, all powered by
  `scipy.stats` (zero new dependencies).
- **Uniform `StatResult` dataclass** — every test returns
  `(test_name, statistic, pvalue, extra)` for a consistent interface.
- **Functions**: two_sample_ttest, paired_ttest, mann_whitney_u,
  one_way_anova, kruskal_wallis (hypothesis); shapiro_test, dagostino_test,
  anderson_test (normality); pearson_correlation, spearman_correlation,
  kendall_correlation, correlation_matrix (correlation); chi_square
  (categorical); describe (descriptive).
- **CLI `stats` subcommand group**: `describe`, `ttest`, `anova`,
  `correlation`, `normality` — for quick exploratory analysis from the
  terminal.
- **20 new tests** (167 total): every stats function tested on synthetic
  data with known expected behavior (significance, correlation strength,
  normality acceptance/rejection).

### Design note
Stats are standalone utilities (NOT Optuna objectives and NOT a TaskType).
They produce p-values/statistics, not values to optimize over trials.

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
