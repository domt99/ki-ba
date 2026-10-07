# KI-BA

Uni project for the course **Bildanalyse mit KI / Machine Learning** (image analysis via AI / ML).

> Project idea: TBD — see [ROADMAP.md](ROADMAP.md) for notes, ideas and next steps.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone git@github.com:domt99/ki-ba.git
cd ki-ba
uv sync              # creates .venv and installs all deps (incl. dev tools)
```

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
docs/        report, slides, figures
ROADMAP.md   shared notes, ideas, todos
```

## Workflow

- Small changes: commit to `main` directly. Bigger features: branch + pull request.
- Clear notebook outputs before committing where possible (keeps diffs small).
- Never commit datasets or model weights — document where to get them instead.
