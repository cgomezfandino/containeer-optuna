"""Command-line interface for containeer-optuna.

Built with Typer. Subcommands:

* ``run`` — execute an experiment YAML end-to-end.
* ``list-models`` — print the registered models, optionally filtered by type.
* ``list-datasets`` — print the datasets registered in datasets.yaml.
* ``describe`` — print the model card (pros/cons, when-to-use) for a model.
* ``dashboard`` — print the ``optuna-dashboard`` command for a storage URL.
* ``init`` — scaffold a new experiment YAML.

Examples:
    containeer-optuna run config/experiments/clustering_optimization.yaml --n-trials 20
    containeer-optuna list-models --type clustering
    containeer-optuna describe kmeans
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ..config import (
    ExperimentConfig,
    get_experiment_configs,
    load_config,
)
from ..evaluation import all_model_cards, get_model_card
from ..optimization import OptunaRunner

app = typer.Typer(
    name="containeer-optuna",
    help="Optuna-first data science framework: run experiments, explore models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to the experiment YAML."),
    n_trials: int | None = typer.Option(
        None, "--n-trials", "-n", help="Override optimization.n_trials."
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable the Optuna progress bar."
    ),
) -> None:
    """Run an experiment end-to-end from a YAML config."""
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/red] {config_path}")
        raise typer.Exit(code=2)

    cfg = load_config(config_path)
    console.print(f"[bold green]Running experiment:[/bold green] {cfg.name}")
    console.print(
        f"  task={cfg.task}  dataset={cfg.dataset}  model={cfg.model}  "
        f"scaler={cfg.scaler}  reducer={cfg.reducer}"
    )

    runner = OptunaRunner(cfg)
    try:
        runner.run(n_trials=n_trials, show_progress_bar=not no_progress)
    except Exception as e:  # pragma: no cover — surface CLI errors clearly
        console.print(f"[red]Run failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    summary = runner.quick_summary()
    table = Table(title=f"Study '{summary['study_name']}' — best trial", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("best_value", f"{summary['best_value']:.4f}")
    for k, v in summary["best_params"].items():
        table.add_row(f"param: {k}", str(v))
    for k, v in summary.get("best_user_attrs", {}).items():
        table.add_row(f"user_attr: {k}", str(v))
    console.print(table)

    console.print(f"\n[dim]Inspect live with:[/dim] optuna-dashboard {cfg.optimization.storage}")


@app.command(name="list-models")
def list_models(
    kind: str | None = typer.Option(
        None, "--type", "-t", help="Filter by kind (regression|clustering|reducer|scaler)."
    ),
) -> None:
    """List registered models and their kind."""
    cards = all_model_cards()
    if kind:
        cards = [c for c in cards if c.kind == kind]
    if not cards:
        console.print(f"[yellow]No models of type '{kind}'.[/yellow]")
        raise typer.Exit()

    table = Table(title="Registered models")
    table.add_column("name", style="cyan")
    table.add_column("kind", style="magenta")
    table.add_column("summary")
    for c in cards:
        table.add_row(c.name, c.kind, c.summary)
    console.print(table)


@app.command(name="list-datasets")
def list_datasets() -> None:
    """List datasets registered in datasets.yaml."""
    from ..config import _base_dir

    config_path = _base_dir() / "config" / "datasets.yaml"
    if not config_path.exists():
        console.print(f"[red]datasets.yaml not found at {config_path}[/red]")
        raise typer.Exit(code=2)

    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f)

    table = Table(title="Registered datasets")
    table.add_column("name", style="cyan")
    table.add_column("source", style="magenta")
    table.add_column("target", style="white")
    for ds in data.get("datasets", []):
        source = (
            f"kaggle:{ds.get('kaggle_dataset')}"
            if ds.get("download")
            else ds.get("path", "(bundled)")
        )
        table.add_row(
            ds.get("name", "?"),
            source,
            ds.get("target_column") or "(none — clustering)",
        )
    console.print(table)


@app.command()
def describe(
    model: str = typer.Argument(..., help="Model registry name (e.g. 'kmeans')."),
) -> None:
    """Print the model card (summary, pros, cons, when-to-use) for a model."""
    card = get_model_card(model)
    if card is None:
        console.print(f"[red]No model card for '{model}'.[/red]")
        console.print("Try `containeer-optuna list-models`.")
        raise typer.Exit(code=2)

    console.print(f"[bold cyan]{card.name}[/bold cyan]  [magenta]({card.kind})[/magenta]")
    console.print(f"\n[bold]Summary:[/bold] {card.summary}")
    console.print(f"\n[bold]When to use:[/bold] {card.when_to_use}")

    if card.pros:
        console.print("\n[bold green]Pros:[/bold green]")
        for p in card.pros:
            console.print(f"  + {p}")
    if card.cons:
        console.print("\n[bold red]Cons:[/bold red]")
        for c in card.cons:
            console.print(f"  - {c}")
    if card.assumptions:
        console.print("\n[bold]Assumptions:[/bold]")
        for a in card.assumptions:
            console.print(f"  • {a}")
    if card.complexity:
        console.print(f"\n[bold]Complexity:[/bold] {card.complexity}")
    if card.key_hyperparameters:
        console.print(f"\n[bold]Key hyperparameters:[/bold] {', '.join(card.key_hyperparameters)}")
    console.print(f"\n[dim]Milestone:[/dim] {card.milestone}")


@app.command()
def dashboard(
    storage: str = typer.Option(
        "sqlite:///optuna_studies.db", "--storage", "-s", help="Optuna storage URL."
    ),
) -> None:
    """Print the optuna-dashboard command for the given storage."""
    console.print(f"[dim]Run:[/dim] optuna-dashboard {storage}")


@app.command()
def init(
    name: str = typer.Argument(..., help="Experiment name (used for file name)."),
    task: str = typer.Option("clustering", "--task", "-t", help="regression | clustering"),
    dataset: str = typer.Option("iris", "--dataset", "-d"),
    model: str = typer.Option("kmeans", "--model", "-m"),
    out_dir: Path = typer.Option(
        Path("config/experiments"), "--out", "-o", help="Output directory."
    ),
) -> None:
    """Scaffold a new experiment YAML."""
    if task not in ("regression", "clustering"):
        console.print(f"[red]task must be 'regression' or 'clustering', got '{task}'[/red]")
        raise typer.Exit(code=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.yaml"

    yaml_text = f"""\
name: {name}
task: {task}
dataset: {dataset}
model: {model}
scaler: standard_scaler
reducer: null
random_state: 42

cv:
  strategy: {"kfold" if task == "clustering" else "shuffle_split"}
  n_splits: {"3" if task == "clustering" else "5"}
  test_size: 0.2
  random_state: 42

optimization:
  enabled: true
  n_trials: 50
  direction: maximize
  storage: sqlite:///{"clustering_optuna" if task == "clustering" else "optuna_studies"}.db
  load_if_exists: true
  sampler: tpe
  pruner: null
"""
    out_file.write_text(yaml_text)
    console.print(f"[green]Created:[/green] {out_file}")


@app.command(name="list-experiments")
def list_experiments() -> None:
    """List all experiment YAMLs in config/experiments/."""
    try:
        configs: list[ExperimentConfig] = get_experiment_configs()
    except Exception as e:  # pragma: no cover
        console.print(f"[red]Failed to load experiments:[/red] {e}")
        raise typer.Exit(code=2) from e

    if not configs:
        console.print("[yellow]No experiments found in config/experiments/.[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("name", style="cyan")
    table.add_column("task", style="magenta")
    table.add_column("dataset", style="white")
    table.add_column("model")
    table.add_column("n_trials", justify="right")
    for c in configs:
        table.add_row(c.name, c.task, c.dataset, c.model, str(c.optimization.n_trials))
    console.print(table)


def main() -> None:
    """Entry point for ``python -m containeer_optuna.cli``."""
    app()


# --- Statistics subcommands (M5) ----------------------------------------

stats_app = typer.Typer(
    name="stats",
    help="Statistical analysis: descriptive stats, hypothesis tests, correlation.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(stats_app, name="stats")


def _load_dataset_for_stats(dataset_name: str) -> pd.DataFrame:
    """Load a dataset as a single DataFrame (features + target if present)."""
    from ..data import get_dataset

    loaded = get_dataset(dataset_name).load()
    if isinstance(loaded, tuple):
        X, y = loaded
        df = X.copy()
        df["_target"] = y
    else:
        df = loaded.copy()
    return df


@stats_app.command(name="describe")
def stats_describe(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. 'iris', 'diabetes')."),
) -> None:
    """Print descriptive statistics for every numeric column."""
    from ..statistics import describe as describe_fn

    df = _load_dataset_for_stats(dataset)
    table = Table(title=f"Descriptive statistics — {dataset}")
    table.add_column("column", style="cyan")
    for stat in ["n", "mean", "std", "min", "q1", "median", "q3", "max", "skew", "kurtosis"]:
        table.add_column(stat, justify="right")

    for col in df.select_dtypes(include=["number"]).columns:
        d = describe_fn(df[col].dropna())
        table.add_row(
            col,
            *[
                f"{d[s]:.3f}"
                for s in [
                    "n",
                    "mean",
                    "std",
                    "min",
                    "q1",
                    "median",
                    "q3",
                    "max",
                    "skew",
                    "kurtosis",
                ]
            ],
        )
    console.print(table)


@stats_app.command(name="ttest")
def stats_ttest(
    dataset: str = typer.Argument(...),
    group_by: str = typer.Option("_target", "--group-by", "-g"),
    feature: str = typer.Option(..., "--feature", "-f", help="Numeric column to test."),
) -> None:
    """Two-sample t-test of a feature across two groups."""
    from ..statistics import two_sample_ttest

    df = _load_dataset_for_stats(dataset)
    if group_by not in df.columns:
        console.print(f"[red]Column '{group_by}' not found.[/red]")
        raise typer.Exit(code=2)
    groups = df.groupby(group_by)[feature].apply(list)
    if len(groups) < 2:
        console.print(f"[red]Need ≥2 groups in '{group_by}'; found {len(groups)}.[/red]")
        raise typer.Exit(code=2)
    result = two_sample_ttest(groups.iloc[0], groups.iloc[1])
    console.print(f"[bold cyan]{result.test_name}[/bold cyan]  ({feature} ~ {group_by})")
    console.print(f"  statistic: {result.statistic:.4f}")
    pval = result.pvalue if result.pvalue is not None else 1.0
    console.print(f"  p-value:   {pval:.6f}")
    if pval < 0.05:
        console.print("  [green]Significant at α=0.05 (reject H₀ of equal means).[/green]")
    else:
        console.print("  [yellow]Not significant at α=0.05.[/yellow]")


@stats_app.command(name="anova")
def stats_anova(
    dataset: str = typer.Argument(...),
    group_by: str = typer.Option("_target", "--group-by", "-g"),
    feature: str = typer.Option(..., "--feature", "-f"),
) -> None:
    """One-way ANOVA of a feature across groups."""
    from ..statistics import one_way_anova

    df = _load_dataset_for_stats(dataset)
    groups = list(df.groupby(group_by)[feature].apply(list))
    if len(groups) < 2:
        console.print(f"[red]Need ≥2 groups; found {len(groups)}.[/red]")
        raise typer.Exit(code=2)
    result = one_way_anova(*groups)
    console.print(f"[bold cyan]{result.test_name}[/bold cyan]  ({feature} ~ {group_by})")
    console.print(f"  F-statistic: {result.statistic:.4f}")
    console.print(f"  p-value:     {result.pvalue:.6f}")
    console.print(
        f"  df:          {result.extra.get('df_between')} / {result.extra.get('df_within')}"
    )


@stats_app.command(name="correlation")
def stats_correlation(
    dataset: str = typer.Argument(...),
    method: str = typer.Option("pearson", "--method", "-m", help="pearson|spearman|kendall"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Min |r| to show."),
) -> None:
    """Print the correlation matrix (features with |r| ≥ threshold)."""
    from ..statistics import correlation_matrix

    df = _load_dataset_for_stats(dataset)
    corr, pval = correlation_matrix(df, method=method)
    table = Table(title=f"Correlation matrix ({method}) — {dataset}  |r| ≥ {threshold}")
    table.add_column("feature A", style="cyan")
    table.add_column("feature B", style="cyan")
    table.add_column("r", justify="right")
    table.add_column("p-value", justify="right")

    cols = corr.columns.tolist()
    shown = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                table.add_row(
                    cols[i],
                    cols[j],
                    f"{r:.3f}",
                    f"{pval.iloc[i, j]:.4f}",
                )
                shown += 1
    if shown == 0:
        console.print(f"[yellow]No feature pairs with |r| ≥ {threshold}.[/yellow]")
    else:
        console.print(table)


@stats_app.command(name="normality")
def stats_normality(
    dataset: str = typer.Argument(...),
    feature: str = typer.Option(..., "--feature", "-f"),
) -> None:
    """Shapiro-Wilk normality test on a feature."""
    from ..statistics import shapiro_test

    df = _load_dataset_for_stats(dataset)
    result = shapiro_test(df[feature].dropna())
    console.print(f"[bold cyan]{result.test_name}[/bold cyan]  ({feature})")
    console.print(f"  W-statistic: {result.statistic:.4f}")
    pval = result.pvalue if result.pvalue is not None else 1.0
    console.print(f"  p-value:     {pval:.6f}")
    if pval < 0.05:
        console.print("  [yellow]Rejects normality at α=0.05 (data is NOT normal).[/yellow]")
    else:
        console.print("  [green]Cannot reject normality at α=0.05 (data may be normal).[/green]")


if __name__ == "__main__":  # pragma: no cover
    main()
