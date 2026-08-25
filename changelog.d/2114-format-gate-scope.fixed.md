- **`ruff format --check` and `ruff check` now cover the repository, not just `mfgarchon/`**
  (Issue #2114). Both gates stopped exactly where most changes land, and the failure is silent —
  an unformatted file produces no signal anywhere in the pipeline.

  It bit twice in one day. `tests/unit/test_check_citations.py` reached `main` unformatted through
  a PR that was green on the local gate **and** green in CI (#2102). Then `tests/conftest.py` was
  made unformatted in #2120 and caught only because an adversarial reviewer ran the formatter over
  a tree the gate does not. The invariant was being held by habit — contributors run `ruff format`
  with no path argument — and habit is what a gate exists to replace.

  **Measured before widening, both were free.** Format: 1 file to reformat out of 936 (that one
  file, whose token stream is byte-identical after reformatting — 3767 tokens either way, so the
  change is line-wrapping only). Lint: `scripts/`, `examples/` and `benchmarks/` are 105 files the
  lint step never saw, with **0 violations between them**. That window was closing, not opening:
  the number was 1 today and is 0 after.

  Verified the widened gate actually fires rather than merely covering more ground — an unformatted
  file planted in `scripts/`, `examples/`, `benchmarks/` and `tests/` turns the gate RED in each
  case, and an unused import in `scripts/` turns the lint step red. Removing them returns it to
  GREEN.

  `ruff` reads its own `exclude` from `pyproject.toml`, so `.` means the repository as the project
  defines it. `.pre-commit-config.yaml` excludes `^investigations/` from `ruff-format`; that
  directory does not exist, so the two are not in conflict.
