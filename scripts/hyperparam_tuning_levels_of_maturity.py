"""
Mirror script for notebooks/optuna/hyperparam_tuning_levels_of_maturity.ipynb

Auto-generated from the notebook code cells.
See the corresponding tutorial in docs/tutorials/ for context.
Original notebook: 8 markdown cells, 30 code cells.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

# %%
df = pd.read_fwf(
    "./data/auto-mpg.data",
    names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
)

df.head()

# %%
cleaned_df = df.copy()

cleaned_df["horsepower"] = cleaned_df["horsepower"].replace("?", np.nan)
cleaned_df["horsepower"] = pd.to_numeric(cleaned_df["horsepower"], errors="coerce")

cleaned_df["car_name"] = cleaned_df["car_name"].str.replace("\"", "")

cleaned_df.dropna(inplace=True)

cleaned_df.head()

# %%
numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cleaned_df[numeric_columns])

scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)

scaled_df

# %%
cv_splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

# %%
selected_features = ["weight", "cylinders"]

# %%
X = scaled_df[selected_features].copy()
y = scaled_df["horsepower"].copy()

# %%
model = Ridge(alpha=2.0, random_state=42)

cv_results = cross_validate(model, X, y, cv=cv_splitter)

# %%
cv_results

# %%
print(f"Average test r2 score: {cv_results['test_score'].mean()}")

# %%
model_1 = Ridge(alpha=1.0, random_state=42)
model_2 = Ridge(alpha=0.5, random_state=42)
model_3 = Ridge(alpha=2.0, random_state=42)

for model in [model_1, model_2, model_3]:

    cv_results = cross_validate(model, X, y, cv=cv_splitter)

    print(f"Average test r2 score: {cv_results['test_score'].mean()}")
    print("--------------")

# %%
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import halfnorm

# %%
params = {
    "alpha": np.linspace(0.5, 5.0, 20),
    "fit_intercept": [True, False],
    "tol": np.linspace(0.0001, 0.001, 20),
    "solver": ["sag", "lsqr"]
}

grid_search = GridSearchCV(
    Ridge(random_state=42),
    param_grid=params,
    cv=cv_splitter,
)
grid_search.fit(X[selected_features], y)

# %%
gs_results_df = pd.DataFrame(grid_search.cv_results_)

gs_results_df.head()

# %%
gs_results_df[gs_results_df["rank_test_score"] == 1]

# %%
len(gs_results_df)

# %%
params = {
    "alpha": halfnorm(loc=2, scale=1), ##np.linspace(0.5, 5.0, 20),
    "fit_intercept": [True, False],
    "tol": np.linspace(0.0001, 0.01, 20),
    "solver": ["sag", "lsqr"]
}

random_search = RandomizedSearchCV(
    Ridge(random_state=42),
    param_distributions=params,
    n_iter=100,
    cv=cv_splitter,
)
random_search.fit(X[selected_features], y)

# %%
rs_results_df = pd.DataFrame(random_search.cv_results_)

rs_results_df

# %%
len(rs_results_df)

# %%
print(random_search.best_score_)

# %%
import optuna
from optuna import Trial

# %%
def objective(trial: Trial):

    model = Ridge(
        alpha = trial.suggest_float("alpha", 0.5, 5.0),
        fit_intercept = False, ##trial.suggest_categorical("fit_intercept", [True, False]),
        tol = trial.suggest_float("tol", 0.0003, 0.0006),
        solver = trial.suggest_categorical("solver", ["lsqr", "sag"]),
        random_state=42,
    )

    cv_results = cross_validate(model, X, y, cv=cv_splitter)

    mean_test_r2 = cv_results["test_score"].mean()

    return mean_test_r2

# %%
study = optuna.create_study(
    study_name="auto_mpg_excl_intercept",
    direction="maximize",
    storage="sqlite:///optuna_studies.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# %%
study.best_value

# %%
study.best_params

# %%
optuna.visualization.plot_param_importances(study)

# %%
optuna.visualization.plot_slice(study, params=["alpha"])

# %%
optuna.visualization.plot_contour(study, params=["alpha","tol"])

# %%

