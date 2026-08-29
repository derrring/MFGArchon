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
  That is now what the paragraph says — **and the imperative is kept**. A first rewrite dropped it,
  called a fresh venv "better", and asserted that `uv venv` "resolves current versions", which is
  false: `uv venv` installs nothing. The same PR body said so two sections earlier. The original was
  false in the same way, and was checked only for whether it was superseded, never for whether it
  was true.
- Regenerating instead of deleting was rejected: the lock is unsatisfied by the current
  `pyproject.toml`, and #2167 restructures that file — dev tooling into `[dependency-groups]`,
  backends into extras with explicit indexes, `[tool.uv] environments` to bound the platform set.
  A lock generated now would be regenerated after every one of those steps, and would meanwhile lock
  universally: measured, a torch-bearing universal lock carries 15 `nvidia-*` packages that a
  darwin-bounded one does not.
- **A guard so it cannot come back as a second pin.** A lock records a ruff version, which is the
  pin living outside its owner and cannot be caught by the scan in
  `test_ruff_pin_has_one_reader_2135.py`, whose population is `scripts/` and `.github/`.
- **Its first version was three defects, all found by adversarial review before merge.** It read the
  pin with a fresh regex — a *second reader*, in the file that exists to enforce one, and one
  matching a bare `ruff-pre-commit` rather than an anchored `- repo:` line, which is exactly the
  #2139 defect already fixed. Demonstrated: a `pre-commit-hooks` block mentioning `ruff-pre-commit`
  in a comment made it return that block's `rev: v6.0.0`. It now calls
  `update_ruff_version.py --print-current`, the reader #2151 added for this. It used `search()`,
  which reads only the first `[[package]]` block, while uv writes one per resolution fork in
  ascending order — measured on a real forked lock, it read 0.16.0 and passed while other platforms
  installed 0.16.5. Now `findall`, and every entry must agree. And the reader was never executed at
  all, because the assertion runs only when a lock exists, which is never in CI; a separate case
  calls it unconditionally.
- **The guard is dormant, and will fire correctly the first time #2167 generates a lock.**
  `pyproject.toml` declares `ruff>=0.6.0` with no upper bound while the pre-commit pin is bumped
  monthly, so a resolver takes the newest and the two agree only on the day of a bump. Measured: a
  real `uv lock` on this tree resolves **0.16.5** against the pinned **0.16.0**. The earlier proof of
  this guard used a hand-written synthetic lock and never ran `uv lock`, so the first real one
  falsified the claim that #2167 would satisfy it. The failure message now names #2172 as the remedy
  instead of "regenerate", which yields 0.16.5 again.
