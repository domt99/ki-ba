# Notes for Claude

## Commits

- Never add Claude as co-author (no `Co-Authored-By: Claude ...` trailer), and no "Generated with
  Claude Code" lines in PR descriptions.
- Credit goes to the two of us only: the `.githooks/prepare-commit-msg` hook adds the other team
  member as `Co-authored-by:`. If a commit lacks that trailer, check that `core.hooksPath` is set
  to `.githooks` and `user.email` is one of the addresses the hook knows (see README Setup).
