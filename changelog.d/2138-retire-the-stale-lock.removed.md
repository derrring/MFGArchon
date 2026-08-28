- **`uv.lock` is no longer tracked** (Issues #2138, #2147). It was five months stale — last touched
  `f1999cc0`, 2026-03-26 — and `uv lock --check` refuses it: *"The lockfile at `uv.lock` needs to be
  updated"*. Four declared runtime dependencies were absent from it (`rich`, `scikit-fem`, `meshio`,
  `osqp`), against 183 locked packages.
- **Nothing read it.** Measured across the whole repository, not a filtered subset: no workflow and
  no script runs `uv sync`. `scripts/setup_development.sh:55` runs `uv venv`, which creates an empty
  environment and does not read a lock. The only references were a `.gitignore` comment asserting it
  was kept "for reproducible research environments", one live `AGENTS.md` instruction, a changelog
  entry, and two synthetic test fixtures.
- **It was not inert, it was harmful.** It pinned ruff 0.13.1 against the `.pre-commit-config.yaml`
  pin of 0.16.0, and pytest 8.4.1 against the gate's 9.1.1. An interpreter carrying that toolchain
  produced `GATE RED` on a two-file documentation diff — six warning identities reported GONE and one
  NEW, one of them `PytestRemovedIn10Warning`, a class pytest 8 cannot emit (#2147). A tracked lock
  that nothing reads is a reproducibility claim with nothing behind it, and this one was actively
  wrong.
- **The `AGENTS.md` instruction it supported is rewritten rather than dropped.** *"Do not build the
  worktree a fresh `uv venv`: `uv.lock` is tracked … and pins exactly that toolchain"* was live — the
  `[SUPERSEDED 2026-08-28]` tag a few lines below belongs to a different paragraph, about porting a
  per-script refusal. The hazard it names is the *activated venv*, which satisfies the gate's probe
  in full so nothing about the selection looks wrong; the lock only made that failure deterministic.
  That is now what the paragraph says.
- Regenerating instead of deleting was rejected: the lock is unsatisfied by the current
  `pyproject.toml`, and #2167 restructures that file — dev tooling into `[dependency-groups]`,
  backends into extras with explicit indexes, `[tool.uv] environments` to bound the platform set.
  A lock generated now would be regenerated after every one of those steps, and would meanwhile lock
  universally: measured, a torch-bearing universal lock carries 15 `nvidia-*` packages that a
  darwin-bounded one does not.
- **A guard so it cannot come back as a second pin.** `uv.lock` records a ruff version, which is the
  pin living outside its owner — and unlike the call sites `test_ruff_pin_has_one_reader_2135.py`
  already scans, it cannot be caught by looking through `scripts/` and `.github/`. The new case
  belongs in that file because a lock carrying a different ruff *is* a second owner for that pin. It
  asserts nothing when no lock is present, which is why the extractor is first made to find a version
  in a synthetic lock that has one: a vacuous assertion passes just as loudly when the extractor is
  broken. Proven in both directions — a lock recording 0.13.1 fails with the version pair named, a
  lock recording 0.16.0 passes. This is what #2167 will have to satisfy when it generates a current
  lock.
