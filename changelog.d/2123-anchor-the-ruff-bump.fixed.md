- **The monthly ruff bump rewrote every hook's pin, not just ruff's** (Issue #2123).
  `check-ruff-updates.yml` ran `sed -i "s/rev: v[0-9.]\+/rev: v$LATEST/" .pre-commit-config.yaml`
  with nothing tying it to the ruff repo. That file carries **two** `rev:` lines —
  `astral-sh/ruff-pre-commit` at v0.16.0 and `pre-commit/pre-commit-hooks` at v6.0.0 — and `s///`
  applies once **per line**, not per file. Simulated at `LATEST = 0.17.0`, the expression set
  `pre-commit-hooks` to `v0.17.0`, a tag that does not exist, so `pre-commit` could fetch no hook
  environment at all: not a formatting failure but a total hook failure, in the bot's own PR, whose
  entire purpose is a routine version bump.

  It had not fired because the step only runs on `needs_update` and no bot PR has merged since that
  second `rev:` line was added. The next successful bump triggers it.

  The `sed` is now anchored to the ruff repo with a range address, and **carries a guard**, because
  an anchored expression comes unanchored again in one edit and nothing downstream would notice — a
  broken pin fails at `pre-commit install-hooks`, in the PR, not in the job that wrote it:

  ```
  changed=$(git diff --numstat .pre-commit-config.yaml | awk '{print $1}')
  [ "$changed" = "1" ] || { echo "::error::..."; git diff ...; exit 1; }
  ```

  **The second `sed` in that step was dead, and so was its twin in `scripts/update_ruff_version.py`.**
  Both rewrote a `ruff==` line in `.github/workflows/modern_quality.yml`. That line moved out; the
  file now says *"Ruff formatting and linting (covered by ci.yml quick-checks)"* and contains
  `ruff==` **zero** times. `ci.yml` holds no pin either — it **reads** the version out of
  `.pre-commit-config.yaml` at runtime (`ci.yml:79`). So the pin has exactly one owner and a bumper
  that touches anything else is how an owner stops being one. Both are removed.

  `update_ruff_version.py`'s own regex was already anchored and had nothing asserting it: measured
  with controls, `tests/` mentioned `update_ruff_version` **0** times and `.pre-commit-config.yaml`
  **0** times (controls: `check_warnings` 1 file, `ci_markers` 4). It now has
  `test_ruff_pin_bump_touches_one_owner_2123.py`, which pins the bump against the fixture **and**
  against the repository's real config so the two cannot diverge, names the retired unanchored
  expression as the counterexample rather than only asserting the fix, and checks that a decoy
  `modern_quality.yml` is left untouched. Reverting the anchor reddens 4 of 7.

  Also corrected while in the file: `update_files` was annotated `-> None` and has always returned
  the list of paths it wrote — the annotation is now `list[str]`, which is what the new test asserts
  against.
