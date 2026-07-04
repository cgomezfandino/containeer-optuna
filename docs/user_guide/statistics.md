# Statistics

The `statistics/` subpackage provides hypothesis tests, normality tests,
correlation, chi-square, and descriptive statistics — all powered by
`scipy.stats` (already a dependency, zero new deps). Stats are standalone
utilities, not Optuna objectives.

## Uniform return type

Every test returns a `StatResult`:

```python
from containeer_optuna import StatResult, two_sample_ttest

r = two_sample_ttest(group_a, group_b)
# StatResult(test_name="Two-sample t-test", statistic=-2.0, pvalue=0.04,
#            extra={"df": 198})
```

## Available tests

### Hypothesis tests
- `two_sample_ttest(a, b)` — independent t-test
- `paired_ttest(a, b)` — paired t-test
- `mann_whitney_u(a, b)` — non-parametric alternative to t-test
- `one_way_anova(*groups)` — one-way ANOVA (≥2 groups)
- `kruskal_wallis(*groups)` — non-parametric alternative to ANOVA

### Normality
- `shapiro_test(x)` — Shapiro-Wilk
- `dagostino_test(x)` — D'Agostino K²
- `anderson_test(x, dist="norm")` — Anderson-Darling (critical values, no p-value)

### Correlation
- `pearson_correlation(x, y)` — linear correlation
- `spearman_correlation(x, y)` — rank correlation (monotonic)
- `kendall_correlation(x, y)` — Kendall's tau
- `correlation_matrix(df, method="pearson")` — full matrix + p-values

### Categorical
- `chi_square(contingency_table)` — chi-square test of independence

### Descriptive
- `describe(data)` — mean, std, quartiles, skew, kurtosis, n

## CLI

```bash
containeer-optuna stats describe iris
containeer-optuna stats ttest iris_classification --group-by _target --feature sepal_length
containeer-optuna stats anova iris_classification --group-by _target --feature sepal_length
containeer-optuna stats correlation diabetes --method pearson --threshold 0.3
containeer-optuna stats normality diabetes --feature bmi
```
