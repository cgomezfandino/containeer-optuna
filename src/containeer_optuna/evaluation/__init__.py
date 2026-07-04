"""Evaluation: metrics, model cards, and visualization.

Public API:
    REGRESSION_SCORERS, get_regression_scorer
    regression_metrics, clustering_metrics
    ModelCard, get_model_card, all_model_cards, card_to_dict
    plot_embedding_2d, plot_scree
"""

from .metrics import (
    REGRESSION_SCORERS,
    clustering_metrics,
    get_regression_scorer,
    regression_metrics,
)
from .model_cards import (
    ModelCard,
    all_model_cards,
    card_to_dict,
    get_model_card,
)
from .plotting import plot_embedding_2d, plot_scree

__all__ = [
    "REGRESSION_SCORERS",
    "get_regression_scorer",
    "regression_metrics",
    "clustering_metrics",
    "ModelCard",
    "get_model_card",
    "all_model_cards",
    "card_to_dict",
    "plot_embedding_2d",
    "plot_scree",
]
