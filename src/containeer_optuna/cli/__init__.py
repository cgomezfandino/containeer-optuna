"""Command-line interface for containeer-optuna.

Exposes the Typer ``app`` so the entry point ``containeer_optuna.cli:app``
declared in pyproject.toml resolves.
"""

from .cli import app, main

__all__ = ["app", "main"]
