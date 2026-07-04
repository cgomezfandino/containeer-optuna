# Statistics tutorial

A typical exploratory data analysis (EDA) workflow using the statistics module.

## 1. Descriptive statistics

```python
from containeer_optuna import get_dataset, describe

X, y = get_dataset("diabetes").load()
print(describe(X["bmi"]))
```

Or from the CLI:

```bash
containeer-optuna stats describe diabetes
```

## 2. Normality check

```python
from containeer_optuna import shapiro_test

r = shapiro_test(X["bmi"])
print(f"W={r.statistic:.4f}, p={r.pvalue:.4f}")
# p < 0.05 → reject normality
```

## 3. Group comparison (t-test)

Compare a feature across two classes:

```bash
containeer-optuna stats ttest breast_cancer --group-by _target --feature "mean radius"
```

## 4. One-way ANOVA

Compare a feature across ≥2 groups:

```bash
containeer-optuna stats anova iris_classification --group-by _target --feature sepal_length
# F=119.26, p=0.000 → significant difference between species
```

## 5. Correlation matrix

Find highly-correlated features:

```bash
containeer-optuna stats correlation diabetes --method pearson --threshold 0.4
```

## 6. Chi-square (categorical)

```python
import numpy as np
from containeer_optuna import chi_square

table = np.array([[90, 10], [10, 90]])
r = chi_square(table)
print(f"chi2={r.statistic:.2f}, p={r.pvalue:.6f}")
```
