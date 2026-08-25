- **The local gate's `ruff format --check` and `ruff check` now cover the repository, not just
  `mfgarchon/`** (Issue #2114), and CI's format check follows. Both gates stopped exactly where most
  changes land, and the failure is silent — an unformatted file produces no signal anywhere in the
  pipeline. CI's *lint* step is a different, narrower check (`ruff check --select F mfgarchon/`, the
  syntax-and-undefined-names tier) and is untouched.

  It bit twice in one day. `tests/unit/test_check_citations.py` reached `main` unformatted through
  a PR that was green on the local gate **and** green in CI (#2102). Then `tests/conftest.py` was
  made unformatted in #2120 and caught only because an adversarial reviewer ran the formatter over
  a tree the gate does not. The invariant was being held by habit — contributors run `ruff format`
  with no path argument — and habit is what a gate exists to replace.

  **Measured before widening, both were free.** Format: 1 file to reformat out of 936. Its reformat
  is line-wrapping only — no `STRING`, `COMMENT` or `FSTRING_MIDDLE` token changed, and the token
  sequence with layout tokens (`NL`, `NEWLINE`, `INDENT`, `DEDENT`, `COMMENT`) dropped is identical
  at 3767 either way. The raw stream is *not* identical (4465 → 4459); it is exactly the layout
  tokens a reformat is supposed to move. Lint: **124 files the lint step never saw**, with **0
  violations between them** under the configured `per-file-ignores` — 105 `.py` under `scripts/`,
  `examples/` and `benchmarks/`, plus 11 notebooks under `examples/`, 7 `.py` under
  `.github/scripts/`, and `pyproject.toml` itself. That window was closing, not opening: the format
  number was 1 today and is 0 after.

  Verified the widened gate actually fires rather than merely covering more ground — an unformatted
  file planted in `scripts/`, `examples/`, `benchmarks/` and `tests/` turns the gate RED in each
  case, and an unused import in `scripts/` turns the lint step red. Removing them returns it to
  GREEN.

  **What `.` means here is four filters, not one.** Ruff's default file-type filter walks `.py`,
  `.pyi`, `.ipynb`, plus `.md` for `format` and `pyproject.toml` for `check` — 46 of 1253 tracked
  files never reach the gate at all. `[tool.ruff] exclude` drops `archive build dist .venv _build
  buck-out .eggs .tox .mypy_cache .git`, and a pattern with no slash matches a basename at any
  depth, so `archive` also drops `docs/archive/`. `respect-gitignore` (on by default) is what skips
  `venv/ .nox/ .ruff_cache/ .ipynb_checkpoints/` — none of them appear in that list. And
  `[tool.ruff.format] exclude = ["*.md"]` accounts for 268 files on its own: 1204 walked without
  it, 936 with.

  **All four are walk-only, for two different reasons, and only one of them has a setting.**
  `force-exclude` is unset, so the two `exclude` settings do not apply to a path named explicitly on
  the command line. The other two are bypassed by ruff's general rule that a named path is always
  processed, which no setting turns off — measured under `force-exclude = true`, `archive/x.py` and
  a repo `.md` are then refused, while a gitignored `.nox/x.py` is still formatted.

  So `ruff format --check $(git diff --name-only)` is a trap that `force-exclude` does not close,
  and it is worse than markdown: ruff accepts an explicitly named `.yml`, `.json` or `.sh` and
  **parses it as Python**. `a:   1` reformats as an annotated assignment; `{"a":  1}` as a dict
  literal.

  Two consequential sites are updated with the gates rather than left to diverge. The monthly
  `check-ruff-updates.yml` repaired only `mfgarchon/` after bumping the pin, so the first ruff
  release that reflows anything under `tests/ examples/ scripts/ benchmarks/ .github/` would have
  opened the bot's own PR red; it now repairs `.`. And `.pre-commit-config.yaml` excluded
  `^investigations/` from `ruff-format`. That directory was created and deleted in one week in
  November 2025 (`c1195707`, `84b168bc`), and the exclusion was added by the commit that created
  it, so it outlived its subject by nine months — harmless while ruff only walked `mfgarchon/`, a
  live disagreement with the gate the moment anyone recreates the name. Both `exclude:` lines are
  removed.
