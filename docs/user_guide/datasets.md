# Datasets

Datasets are registered in `config/datasets.yaml`. Each entry has:

- `name` — registry key
- `path` (local) or `kaggle_dataset` + `download: true` (Kaggle)
- `target_column` (None for clustering)
- `feature_columns` (optional subset)
- `preprocessing` — recipe dict

## Built-in datasets

| Name | Source | Task |
|------|--------|------|
| `auto_mpg` | local `./data/auto-mpg.data` | regression (target: `horsepower`) |
| `iris` | sklearn bundled | clustering |
| `customer_personality` | Kaggle `imakash3011/customer-personality-analysis` | clustering |
| `credit_card` | Kaggle `arjunbhasin2013/ccdata` | clustering |

## Loading in Python

```python
from containeer_optuna import get_dataset

X = get_dataset("iris").load()              # clustering → returns X
X, y = get_dataset("auto_mpg").load()       # regression → returns (X, y)
```

## Preprocessing recipe

Recognized keys under `preprocessing:`:

- `sep` — separator (`"\\s+"`, `"\\t"`, `,`, ...)
- `na_values` — tokens treated as NaN (e.g. `["?"]`)
- `names` — explicit column names
- `header` — `null` for no header row
- `drop_columns` — columns to drop
- `numeric_conversion` — columns to coerce to numeric
- `one_hot_encode` — categorical columns to one-hot
- `dropna` — drop rows with NaN if true
