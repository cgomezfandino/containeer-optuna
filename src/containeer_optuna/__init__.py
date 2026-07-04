"""containeer-optuna: an Optuna-first data science framework.

A modular framework for running Optuna hyperparameter optimization experiments
across regression, clustering, dimensionality reduction, and (in future
milestones) classification, statistics, and deep learning. Configured via YAML,
driven by a typed (pydantic) config layer, and runnable from the CLI.

Quick start:
    >>> from containeer_optuna import load_config, OptunaRunner
    >>> cfg = load_config("config/experiments/clustering_optimization.yaml")
    >>> study = OptunaRunner(cfg).run(n_trials=20)
    >>> study.best_value, study.best_params

Public API:
    load_config, ExperimentConfig, load_dataset_config, load_model_config,
    get_experiment_configs, settings,
    BaseDataset, get_dataset,
    BaseModel, get_model, suggest_params,
    BasePipeline, get_pipeline,
    OptunaRunner, make_regression_objective, make_clustering_objective,
    regression_metrics, clustering_metrics,
    ModelCard, get_model_card, all_model_cards.
"""

__version__ = "0.1.0"
__author__ = "Carlos Gomez"
__email__ = "cgomezfandino@example.com"

from .config import (
    CVConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    get_experiment_configs,
    load_config,
    load_dataset_config,
    load_model_config,
    settings,
)
from .data import BaseDataset, YamlDatasetLoader, get_dataset
from .evaluation import (
    REGRESSION_SCORERS,
    ModelCard,
    all_model_cards,
    card_to_dict,
    clustering_metrics,
    get_model_card,
    get_regression_scorer,
    plot_embedding_2d,
    plot_scree,
    regression_metrics,
)
from .models import MODEL_CLASSES, BaseModel, get_model, suggest_params
from .optimization import (
    OptunaRunner,
    make_clustering_objective,
    make_cv_splitter,
    make_model_selection_objective,
    make_regression_objective,
)
from .pipelines import BasePipeline, get_pipeline
from .utils import (
    ensure_dir,
    get_logger,
    save_json,
    save_predictions,
    seed_all,
    study_summary,
)

__all__ = [
    # version / metadata
    "__version__",
    # config
    "CVConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OptimizationConfig",
    "load_config",
    "load_dataset_config",
    "load_model_config",
    "get_experiment_configs",
    "settings",
    # data
    "BaseDataset",
    "YamlDatasetLoader",
    "get_dataset",
    # models
    "BaseModel",
    "MODEL_CLASSES",
    "get_model",
    "suggest_params",
    # pipelines
    "BasePipeline",
    "get_pipeline",
    # optimization
    "OptunaRunner",
    "make_cv_splitter",
    "make_regression_objective",
    "make_clustering_objective",
    "make_model_selection_objective",
    # evaluation
    "regression_metrics",
    "clustering_metrics",
    "REGRESSION_SCORERS",
    "get_regression_scorer",
    "ModelCard",
    "get_model_card",
    "all_model_cards",
    "card_to_dict",
    "plot_embedding_2d",
    "plot_scree",
    # utils
    "seed_all",
    "get_logger",
    "ensure_dir",
    "save_json",
    "save_predictions",
    "study_summary",
]
