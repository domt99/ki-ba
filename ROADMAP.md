# Roadmap & Notes

Shared space for status, todos and decisions. Newest entries on top within each section.
Tag entries with your initials (e.g. `[DT]`).

Project: **Projekt 1 – Von XOR zu MNIST** (KI-BA, Prof. Liebscher), MLP in pure numpy.
Task sheet: `docs/01_KI-BA_P (1).pdf` (sub-tasks (i)–(vi) = milestones 1–6).

## Status (2026-11-22)

**Done**
- Optimizers: gradient descent, Rprop−, Adam (`optimizer.py`); activations, layers,
  loop and matrix calculation, MLP, DataPrep (50 000 train / 10 000 validation split)
- Framework: .env/template config + checks, factories, MNIST loading, training with
  accuracy per epoch and early stopping (best validation accuracy, weights restored),
  confusion matrix, misclassified images, `count_parameters`
- `main.py`: `--template <file>` overrides `TEMPLATE` in .env, test data only with
  `--test`; every run writes `result.json` + plots to
  `results/<date>_<hh_mm_ss>_<initials>_<template>/` (initials from `git config user.email`)
- `compare.py`: table of all runs per data set (validation only), `--markdown` for the report
- 6a: learning curves (loss, train/val accuracy, best epoch) for 784-30-10
- 6b runs done (adam, 10 epochs, seed 42), see `uv run python compare.py`:
  784-300-10 97.86 %, 784-128-64-10 97.68 %, 784-100-10 97.51 %,
  784-256-128-64-10 97.50 %, 784-30-10 96.28 % (val)
- Rprop template: full batch (`BATCH_SIZE=50000`), 100 epochs, `DELTA_NULL=0.0125` → ~96.3 % val
- Scaffolds with `# TODO` for 3b, 3c, 4c in `tests/` (skipped until filled in) and
  `scripts/benchmark_loop_vs_matrix.py` for 4a/4d (complete, ~110× faster with matrix)

## Todo (in this order)

- [ ] **First, next session:** we don't understand the test scaffolds Claude prepared on
      2026-11-22 yet (`tests/helpers.py`, `tests/test_xor_known_solution.py`,
      `tests/test_gradient_check.py`, `tests/test_loop_vs_matrix.py`,
      `scripts/benchmark_loop_vs_matrix.py`, commit `1396e44`). Decide together: either revert
      them, or walk through them step by step with Claude until we understand them
      (pytest basics, fixtures/parametrize, why tests are skipped, what each TODO expects).
      Only then continue with 3b/3c/4c.
- [ ] **3b** `tests/test_xor_known_solution.py`: enter values from slides "XOR-MLP: vorwärts"
      and "XOR-MLP: rückwärts" (not in the task PDF – we need the lecture slides), fill in
      the comparisons and the one-step test (o5 ≈ 0.08, o6 ≈ 0.92 after η = 0.5)
- [ ] **3c** `tests/test_gradient_check.py`: `numerical_gradient`, `relative_error`, loop over
      all weights (h = 1e-6, relative error < 1e-6, mind the ReLU kink at 0)
- [ ] [DT] "loop" option = everything with loops, not only the three methods in
      `CalculateLoop`. Still vectorized today:
  - activation functions + derivatives (`activation_function.py`: `np.where`, `np.tanh`, ...)
  - delta = backprop error · f'(net) in `HiddenLayer.calculate_delta` (`layer.py`)
  - bias column (`np.hstack`) in `forward_step`
  - softmax and categorical cross-entropy (`layer.py`)
  - optimizer weight update (`optimizer.py`)? – decide whether it belongs to the comparison
- [ ] [DT] Runtime comparison matrix vs. loop for the full forward + backward step
      (prerequisite for 4a/4d below)
- [ ] **4c** `tests/test_loop_vs_matrix.py`: compare outputs + gradients with `np.allclose`
- [ ] **4a/4d** run `uv run python scripts/benchmark_loop_vs_matrix.py` on our own laptops,
      put the numbers in the report
- [ ] **3d/3e** XOR: epochs until all 4 instances are correct (GD vs. Adam/Rprop, with a max
      epoch count); 20 runs each with 2 and 4 hidden neurons and different seeds – success
      rate, why 2 hidden neurons often fail. Infrastructure (seed loop script) can be done by Claude
- [ ] **6b** write the interpretation (overfitting vs. parameter count, best epoch)
- [ ] **6c** pick the best net by validation only, then run once with `--test`
      (accuracy, confusion matrix, misclassified images, which digits get confused)
- [ ] **6d** compare with literature values from (v), what could improve it
- [ ] Decide: commit the task PDF (maybe renamed to `docs/projekt1-aufgabe.pdf`)?
- [ ] ruff cleanup (postponed): import sorting (`ruff check --fix`), lines > 100,
      unused loop variable in `dataprep.py`, `zip(..., strict=True)` in
      `calculate_confusion_matrix`. Touches our required-part files – ask first
- [ ] [DT] Idea: one notebook per milestone in `notebooks/` (e.g. `milestone_3.ipynb`) that
      walks through the sub-task, runs the code and shows the plots/numbers for the report;
      check how it relates to the existing `milestones/milestone_1.py`, `milestone_2.py`
- [ ] [DT] Extra: classify a single image with an existing model
  - save weights with every run (e.g. `weights.npz` in the run folder via
    `get_weights`/`set_weights`), so a run can be loaded again
  - script e.g. `scripts/predict.py --run <folder> --image <png>`: grayscale, 28×28,
    scaled like DataPrep (and inverted if needed: MNIST is white on black), prints the
    predicted digit + softmax probabilities
  - file size: 784-300-10 ≈ 240k parameters ≈ 2 MB per run – decide whether weights
    are committed or go in `.gitignore`
- [ ] Extra: dashboard for `results/` (own dependency group if it needs any). A mock was
      shown on 2026-11-22 (not in the repo; liked it). Rough scope from that mock:
  - HTML/JS page with a small local server (localhost only), data = all
    `results/*/result.json`
  - [DT] start runs from the dashboard: pick a template/settings, the server starts
    `main.py` in the background (never with `--test`), shows progress and puts the new run
    straight into the charts/table for comparison
  - [DT] if the single-image todo is done: upload/draw a digit and classify it with a
    chosen run
  - filters in one row: data set, optimizer, person (dt/mm); light/dark theme
  - KPI tiles: number of runs, best val accuracy (topology, epoch), smallest net ≥ 97 % val,
    test evaluation status (6c)
  - charts: val accuracy per epoch for all runs (hover crosshair, legend highlight);
    parameters (log x) vs. best val accuracy; learning curves (train/val + best epoch) and
    loss of the selected run
  - sortable comparison table like `compare.py` plus a "train − val" gap column,
    no test accuracy; click a row or point to select a run; color follows the run

## Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-11-22 | Test data only via `main.py --test`, once, for the chosen config | task 6b/6c: choose by validation only |
| 2026-11-22 | Every run → `results/…/result.json` + plots, committed to git | both of us see all runs; files are a few KB |
| 2026-11-22 | Run folder name `<date>_<hh_mm_ss>_<dt/mm>_<template>` | unique per person and second; full config is in the JSON |
| 2026-11-22 | Rprop initial step via `DELTA_NULL` key (default stays 0.5) | 0.5 → ~66 % val, 0.0125 (Igel & Hüsken 2003) → ~96 % |
| 2026-11-22 | XOR template uses sigmoid_log | relu gets stuck at 75 % with seed 42 (dead neuron) |
| 2026-10-07 | Python 3.12 + uv for env management | reproducible, fast, one lockfile for both of us |
