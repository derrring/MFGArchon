- **The local gate's `ruff format --check` and `ruff check` cover the repository, not just
  `mfgarchon/`** (Issue #2114), and CI's format check follows. Both stopped exactly where most
  changes land, and the failure is silent — an unformatted file produced no signal anywhere in the
  pipeline. It bit twice in one day: one file reached `main` unformatted through a PR that was green
  on the local gate *and* green in CI, and a second was caught only because a reviewer ran the
  formatter over a tree the gate did not.

  CI's *lint* step is a different, narrower tier (`ruff check --select F mfgarchon/`) and is
  untouched.

  **What `.` means is four filters, not one**: ruff's default file-type walk, `[tool.ruff] exclude`,
  `respect-gitignore`, and `[tool.ruff.format] exclude`. `force-exclude` is unset, so the two
  `exclude` settings do not bind a path named explicitly on the command line — and the other two are
  bypassed because ruff always processes a named path. So `ruff format --check $(git diff
  --name-only)` is a trap: ruff parses any named file as Python, and the ones that parse
  *successfully* are rewritten in silence.

  Two adjacent sites are updated with the gates rather than left to diverge: the monthly
  `check-ruff-updates.yml` repaired only `mfgarchon/` after bumping the pin, and
  `.pre-commit-config.yaml` excluded a directory that no longer exists.
