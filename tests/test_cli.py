"""Smoke tests for the CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from containeer_optuna.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "describe" in result.stdout


def test_list_models():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "kmeans" in result.stdout
    assert "ridge" in result.stdout


def test_list_models_filtered():
    result = runner.invoke(app, ["list-models", "--type", "clustering"])
    assert result.exit_code == 0
    assert "kmeans" in result.stdout
    # Regression models should NOT appear when filtering to clustering
    assert "ridge" not in result.stdout


def test_describe_known_model():
    result = runner.invoke(app, ["describe", "dbscan"])
    assert result.exit_code == 0
    assert "DBSCAN" in result.stdout or "Density" in result.stdout
    assert "Pros" in result.stdout
    assert "Cons" in result.stdout


def test_describe_unknown_model():
    result = runner.invoke(app, ["describe", "not_a_model"])
    assert result.exit_code == 2


def test_list_datasets():
    result = runner.invoke(app, ["list-datasets"])
    assert result.exit_code == 0
    assert "auto_mpg" in result.stdout
    assert "iris" in result.stdout


def test_list_experiments():
    result = runner.invoke(app, ["list-experiments"])
    assert result.exit_code == 0
    # Names may be truncated by Rich tables; check for a known prefix and task.
    assert "clustering" in result.stdout
    assert "classification" in result.stdout


def test_init_creates_yaml(tmp_path: Path):
    out = tmp_path / "exp"
    result = runner.invoke(
        app,
        [
            "init",
            "my_test",
            "--task",
            "regression",
            "--dataset",
            "auto_mpg",
            "--model",
            "ridge",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert (out / "my_test.yaml").exists()


def test_dashboard():
    result = runner.invoke(app, ["dashboard", "--storage", "sqlite:///x.db"])
    assert result.exit_code == 0
    assert "optuna-dashboard" in result.stdout
