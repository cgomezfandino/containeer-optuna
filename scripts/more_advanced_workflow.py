"""
Mirror script for notebooks/optuna/more_advanced_workflow.ipynb

Auto-generated from the notebook code cells.
See the corresponding tutorial in docs/tutorials/ for context.
Original notebook: 4 markdown cells, 19 code cells.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
from optuna import Trial

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
selected_features = ["weight", "cylinders", "mpg", "displacement", "acceleration"]

# %%
MODEL_DICT = {
    "ridge": Ridge,
    "lasso": Lasso,
    "ols": LinearRegression,
}

# %%
X = cleaned_df[selected_features].copy()
y = cleaned_df["horsepower"].copy()

cv_splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

def model_selection_objective(trial: Trial):

    model_type = trial.suggest_categorical("model_type", ["ridge", "lasso", "ols"])
    model = MODEL_DICT[model_type]

    if model_type in ["ridge", "lasso"]:
        alpha = trial.suggest_float("alpha", 0.5, 3.0, step=0.5)
        tol = trial.suggest_float("tol", 0.0001, 0.001, step=0.0001)
        model_instance = model(alpha=alpha, tol=tol, random_state=42)
    else:
        model_instance = model()
    
    scaler = StandardScaler()
    
    model_pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model_instance)
    ])

    cv_results = cross_validate(model_pipeline, X, y, cv=cv_splitter)

    mean_r2_score = cv_results["test_score"].mean()

    return mean_r2_score
    

# %%
study = optuna.create_study(
    study_name="auto_mpg_model_selection",
    direction="maximize",
    storage="sqlite:///optuna_studies",
    load_if_exists=True
)
study.optimize(model_selection_objective, n_trials=100, show_progress_bar=True)

# %%
study.best_params

# %%
study.best_value

# %%
optuna.visualization.plot_slice(study, params=["model_type"])

# %%
FEATURE_SETS = {
    "wide_set": ["weight", "cylinders", "mpg", "displacement", "acceleration"],
    "narrow_set": ["acceleration", "mpg", "cylinders"],
    "moderate_set": ["weight", "acceleration", "mpg", "displacement"]
}

# %%
cv_splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

def feature_set_selection_objective(trial: Trial):

    model_instance = Ridge(
        alpha=trial.suggest_float("alpha", 0.5, 6.0),
        tol=trial.suggest_float("tol", 0.0001, 0.001),
        random_state=42
    )
    
    feature_set_name = trial.suggest_categorical("feature_set_name", ["wide_set", "narrow_set", "moderate_set"])
    feature_set = FEATURE_SETS[feature_set_name]

    X = cleaned_df[feature_set].copy()
    y = cleaned_df["horsepower"].copy()
    
    scaler = StandardScaler()
    
    model_pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model_instance)
    ])

    cv_results = cross_validate(model_pipeline, X, y, cv=cv_splitter)

    mean_r2_score = cv_results["test_score"].mean()

    return mean_r2_score

# %%
fs_study = optuna.create_study(
    study_name="auto_mpg_feature_set_selection_corrected",
    direction="maximize",
    storage="sqlite:///optuna_studies.db",
    load_if_exists=True
)
fs_study.optimize(feature_set_selection_objective, n_trials=100, show_progress_bar=True)

# %%
fs_study.best_params

# %%
fs_study.best_value

# %%
optuna.visualization.plot_slice(study=fs_study, params=["feature_set_name"])

# %%
optuna.visualization.plot_param_importances(fs_study)

# %%
optuna.visualization.plot_contour(study=fs_study, params=["tol", "alpha"])

# %%

