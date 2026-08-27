- **The ruff pin has one reader and one writer** (Issue #2135). One version was read by five
  different expressions across `scripts/local_ci.sh`, `.github/workflows/ci.yml` and
  `.github/workflows/check-ruff-updates.yml`, four of them a hand-rolled `grep -A1
  astral-sh/ruff-pre-commit`, and written by two — `update_ruff_version.py` and the workflow's own
  `sed -i`. They did not agree on what a ruff block may look like: a comment between `repo:` and
  `rev:` is valid YAML that the writer's expression spans and every `-A1` reader misses, so the
  writer could land a bump that the verifiers then read as absent. #2123 had to be fixed twice for
  the same reason, once on each writer, and the second fix exists only because someone noticed.
  `update_ruff_version.py --print-current` is now the single reader — it prints the pin and nothing
  else, before the banner and before any network call — and `--force` is the single writer; all four
  call sites and the workflow's bumper go through them. `tests/unit/test_ruff_pin_has_one_reader_2135.py`
  fails if a hand-rolled reader or a second writer reappears, and covers the comment shape that made
  the two disagree.
