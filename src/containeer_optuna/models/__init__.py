"""Model registry and factory for containeer-optuna.

Public API:
    BaseModel, get_model, suggest_params
"""

from .classes import MODEL_CLASSES
from .registry import BaseModel, get_model, suggest_params

__all__ = ["BaseModel", "get_model", "suggest_params", "MODEL_CLASSES"]
