# Notes for Claude

## Commits

- Never add Claude as co-author (no `Co-Authored-By: Claude ...` trailer), and no "Generated with
  Claude Code" lines in PR descriptions.
- Credit goes to the two of us only: the `.githooks/prepare-commit-msg` hook adds the other team
  member as `Co-authored-by:`. If a commit lacks that trailer, check that `core.hooksPath` is set
  to `.githooks` and `user.email` is one of the addresses the hook knows (see README Setup).

## Way of working (agreed with both of us)

- Required parts of the task (network logic, training, evaluation) we write ourselves. Claude
  gives only "fits / doesn't fit" plus guiding questions there, no solutions – unless we
  explicitly ask for code. Test scaffolds in `tests/` have `# TODO`s that are ours to fill.
- Claude may write directly: .env/config checks, output/plots, infrastructure (`results.py`,
  `compare.py`, scripts), extras.
- Show changes (diff/draft) before saving; test drafts in the scratchpad, not in the repo.
- Keep our variable names; snake_case for methods/attributes, CamelCase for classes; code and
  comments in English. Answer us in German.
- Test data only once at the end (`main.py --test`); compare configurations on validation only.
- Commit/push only when we say so. GitHub Actions budget: see global notes (no workflows yet).

## Where things are

- Status, todo list (in order) and decisions: `ROADMAP.md` – start there.
- Task sheet: `docs/01_KI-BA_P (1).pdf`. How to run: README "Running".
