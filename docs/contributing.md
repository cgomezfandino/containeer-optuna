# Contributing

Contributions are welcome. This project follows a milestone-based roadmap
(see [Roadmap](roadmap.md)).

## Development setup

```bash
git clone https://github.com/cgomezfandino/containeer-optuna.git
cd containeer-optuna
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Branch from `main` (e.g. `feat/m1-trees`).
2. Write code + tests. Every new public function needs a test and a docstring.
3. Run the quality gates locally:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

4. (If docs touched) `mkdocs build --strict` to catch broken references.
5. Open a PR describing what changed and which milestone it advances.

## Adding a new model

1. `config/models.yaml` — add an entry with `type`, `default_params`, `param_space`.
2. `src/containeer_optuna/models/classes.py` — map the name to the sklearn class.
3. `src/containeer_optuna/evaluation/model_cards.py` — add a `ModelCard` (pros/cons).
4. `tests/test_models.py` — add a test that it instantiates.
5. Regenerate the model card docs (the page is built from the registry at docs-build time).

## Style

- Line length 100, double-quoted strings, Google-style docstrings.
- Type annotations are required on public functions (`mypy --disallow-untyped-def`).
- English in all docs and docstrings; code identifiers in English.

## Roadmap contributions

Pick an open milestone and open an issue to discuss scope before coding.
