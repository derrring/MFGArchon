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
  violations between them** — 105 `.py` under `scripts/`, `examples/` and `benchmarks/`, plus 11
  notebooks under `examples/`, 7 `.py` under `.github/scripts/`, and `pyproject.toml` itself. That
  window was closing, not opening: the format number was 1 today and is 0 after.

  Verified the widened gate actually fires rather than merely covering more ground — an unformatted
  file planted in `scripts/`, `examples/`, `benchmarks/` and `tests/` turns the gate RED in each
  case, and an unused import in `scripts/` turns the lint step red. Removing them returns it to
  GREEN.

  **What `.` means here is three mechanisms, not one.** `[tool.ruff] exclude` in `pyproject.toml`
  drops `archive build dist .venv _build buck-out .eggs .tox .mypy_cache .git`; `respect-gitignore`
  (on by default, and the reason `venv/ .nox/ .ruff_cache/ .ipynb_checkpoints/` are skipped —
  none of them appear in that list); and `[tool.ruff.format] exclude = ["*.md"]` keeps the 269
  markdown files ruff would otherwise walk out of the format gate. That last one is walk-only: a repo `.md` named explicitly on the
  command line is still formatted, which would bite anyone who later writes
  `ruff format --check $(git diff --name-only)`.

  Two consequential sites are updated with the gates rather than left to diverge. The monthly
  `check-ruff-updates.yml` repaired only `mfgarchon/` after bumping the pin, so the first ruff
  release that reflows anything under `tests/ examples/ scripts/ benchmarks/ .github/` would have
  opened the bot's own PR red; it now repairs `.`. And `.pre-commit-config.yaml` excluded
  `^investigations/` from `ruff-format` — harmless while ruff only walked `mfgarchon/`, a live
  disagreement with the gate the moment anyone creates that directory. It does not exist, so the
  two `exclude:` lines protected nothing and are removed.
