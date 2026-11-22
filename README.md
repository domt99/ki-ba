# KI-BA

Uni project for the course **Bildanalyse mit KI / Machine Learning** (image analysis via AI / ML).

> Project idea: TBD — see [ROADMAP.md](ROADMAP.md) for notes, ideas and next steps.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone git@github.com:domt99/ki-ba.git
cd ki-ba
uv sync              # creates .venv and installs all deps (incl. dev tools)
git config core.hooksPath .githooks  # auto-adds the other person as co-author
cp .env.example .env # picks the run configuration (templates in configs/)
```

## Running

```bash
uv run python main.py                                # config from .env (TEMPLATE=... or keys)
uv run python main.py --template mnist-30-adam.env   # template from configs/, overrides .env
uv run python main.py --template mnist-adam.env --test  # also evaluate the test data
uv run python compare.py [--markdown]                # table of all MNIST runs (validation only)
```

Compare configurations on the validation data only; use `--test` once, for the final
configuration. Each run writes `result.json` (config, seed, accuracy per epoch, best epoch)
and its plots to `results/<date>_<time>_<initials>_<template>/`. The initials (`dt`/`mm`)
come from `git config user.email`. Results are committed, so both of us see all runs.

## Common commands

```bash
uv run jupyter lab   # notebooks for exploration
uv run pytest        # tests
uv run ruff check .  # lint
uv run ruff format . # format
uv add <package>     # add a dependency (commit pyproject.toml + uv.lock!)
```

## Structure

```
src/ki_ba/   reusable code (data loading, models, training, ...)
notebooks/   exploration & experiments
data/        datasets — local only, gitignored (see data/README.md)
models/      trained weights/checkpoints — local only, gitignored
tests/       tests
docs/        report, slides, figures (MLP architecture: docs/architecture/, task 2a component table: docs/Meilenstein2a-Komponenten.pdf)
ROADMAP.md   shared notes, ideas, todos
```

## Workflow

- Every commit credits both of us: the hook in `.githooks/` adds the other person as
  `Co-authored-by:` (needs the `core.hooksPath` setting from Setup and `user.email` set to the
  address it knows).
- Small changes: commit to `main` directly. Bigger features: branch + pull request.
- Clear notebook outputs before committing where possible (keeps diffs small).
- Never commit datasets or model weights — document where to get them instead.
