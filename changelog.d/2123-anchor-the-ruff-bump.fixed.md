- **The monthly ruff bump rewrote every hook's pin, not just ruff's** (Issue #2123).
  `check-ruff-updates.yml` ran `sed -i "s/rev: v[0-9.]\+/rev: v$LATEST/" .pre-commit-config.yaml`
  with nothing tying it to the ruff repo. That file carries **two** `rev:` lines —
  `astral-sh/ruff-pre-commit` at v0.16.0 and `pre-commit/pre-commit-hooks` at v6.0.0 — and `s///`
  applies once **per line**, not per file. Simulated at `LATEST = 0.17.0`, the expression set
  `pre-commit-hooks` to `v0.17.0`, a tag that does not exist, so `pre-commit` could fetch no hook
  environment at all. Measured against real `pre-commit` 4.6.1: `pre-commit run --all-files` exits
  **3**, and the healthy ruff hook — listed *first* — never runs; naming only that hook fails too,
  because the failure happens during repo initialization, before hook selection. `git push` is then
  blocked for every contributor.

  **And the bot's own PR checks would have been green**, because no workflow runs `pre-commit`. The
  breakage lands on `main` invisibly and surfaces at the next person's push. Worth stating because
  it removes the reflex that CI would have caught it.

  One shape is worse than a hard failure and is not fixed by anything here: ruff is `0.x` today, so
  the bad rev never resolves. At ruff `2.x` and beyond the unanchored expression would set
  `pre-commit-hooks` to a **real, ancient** tag — installs fine, runs different checks, says
  nothing.

  It had not fired because the step only runs on `needs_update` and no bot PR has merged since that
  second `rev:` line was added. The next successful bump triggers it.

  The `sed` is now anchored to the ruff repo with a range address, and **carries a guard that checks
  identity rather than a line count**. A count of 1 is satisfied by two configurations that
  reproduce the original defect with the anchor in place — a ruff block whose `rev:` precedes its
  `repo:`, and a ruff block with no `rev:` at all; in both, the range lands on `pre-commit-hooks`
  and changes exactly one line. Verified: the count-only form passes both, and re-reading the pin
  the way `ci.yml` does rejects both.

  **The second `sed` in that step was dead, and so was its twin in `scripts/update_ruff_version.py`.**
  Both rewrote a `ruff==` line in `.github/workflows/modern_quality.yml`. That line moved out; the
  file now says *"Ruff formatting and linting (covered by ci.yml quick-checks)"* and contains
  `ruff==` **zero** times. `ci.yml` holds no pin either — it **reads** the version out of
  `.pre-commit-config.yaml` at runtime — the `RUFF_VERSION=$(grep …)` line in `ci.yml`'s
  `quick-checks` job, cited by symbol because a line number in prose expires. So the pin has
  exactly one owner and a bumper
  that touches anything else is how an owner stops being one. Both are removed.

  `update_ruff_version.py`'s own regex was already anchored and had nothing asserting it: measured
  with controls, `tests/` mentioned `update_ruff_version` **0** times and `.pre-commit-config.yaml`
  **0** times (controls: `check_warnings` 1 file, `ci_markers` 4). It now has
  `test_ruff_pin_bump_touches_one_owner_2123.py`, which pins the bump against the fixture **and**
  against the repository's real config so the two cannot diverge, names the retired unanchored
  expression as the counterexample rather than only asserting the fix, and checks that a decoy
  `modern_quality.yml` is left untouched. Reverting the anchor reddens 4 of 7.

  **Two further defects the review found in the Python bumper, both fail-silent.** A comment between
  the `repo:` line and its `rev:` is valid YAML, and `\s+` cannot span it — so the substitution
  matched nothing, `update_files` returned `[]`, and `main()` printed *"No files needed updating"*
  and exited 0, while the workflow's `sed` handled the same file correctly. The pattern now tolerates
  comment lines, and `update_files` **checks its own postcondition**: `main()` only calls it when the
  versions differ, so "nothing changed" is always a defect and never a no-op, and it now raises with
  the shape named. Each of the three has its own pin — reverting the anchor reddens 6 of 10, deleting
  the postcondition 2, narrowing the pattern back to `\s+` 1.

  **The blocker this nearly shipped.** `update_ruff_version.py` imported `requests` at module scope,
  and `requests` is in neither `pyproject.toml` nor `environment.yml` (control: `scipy` is in both) —
  it reaches this environment only through conda and Sphinx. The new test file was the first thing to
  import that module, so collection failed: measured, `6593 collected, 1 error` on the branch against
  a clean `6593` on base, which reddens `nightly (unit)` on every run and leaves the release coverage
  job — which has no `-n` — running **zero** tests. The local gate stayed green only because this
  environment has `requests` by accident. The import is now inside the two network functions;
  verified with a control that the module loads with `requests` blocked and fails to load when the
  import is moved back.

  Also corrected: `update_files` was annotated `-> None` and has always returned the list of paths it
  wrote. Three citations of the `ci.yml` line that reads the pin gave three different numbers, none
  correct; they now name the `RUFF_VERSION=$(grep …)` line by symbol. And the bot's PR body still
  advertised *"Updated `modern_quality.yml`"* — a step that is now deleted and was a no-op before
  that — so every future bot PR would have claimed an update that never happened.

  **"Exactly one owner" is qualified.** `uv.lock` materialises `ruff 0.13.1` against a pin of
  `0.16.0` — already drifted by three minors. Nothing reads it (`uv sync` appears zero times;
  `setup_development.sh` uses `uv pip install -e`, which ignores the lock), so it is dormant rather
  than live, but `.gitignore:86` keeps it tracked "for reproducible research environments" and that
  claim no longer holds. Filed separately.
