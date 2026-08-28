- **`environment.yml` did not declare eight packages the library imports, and still carried three
  that #1687 removed from `pyproject.toml`.** An environment built from it alone cannot run the
  package: measured 2026-08-28, 6630 test outcomes against 6724, with 66 tests skipped or not
  collected because cvxpy and torch were absent — and the warning ratchet then reported the warnings
  those tests would have emitted as identities GONE, inviting the reader to record the loss as
  progress. #1687 did this reconciliation on the criterion "zero imports in the package, the tests,
  the examples and the benchmarks", and did it for one manifest only.
- Added, each because the package imports it: `cvxpy`, `scikit-fem`, `meshio`, `networkx`, `osqp`,
  `nbformat`, `pyyaml`, `rich`. Added because the gate needs them: `pytest-xdist` (`-n auto`),
  `pytest-timeout` (`pytest.ini` sets `timeout = 900` under `--strict-config`, so without it
  collection fails with "Unknown config option"), `pytest-cov`.
- Removed, on #1687's own criterion and evidence: `jupyter`, `jupyterlab`, `seaborn` (already gone
  from `pyproject.toml` for exactly this reason), `ipython`, `texttable` (whose only other trace is
  a `[tool.mypy]` override the gate already reports as unused), and `tqdm` — whose comment claimed
  "used throughout package" while `mfgarchon/utils/progress.py` supplies a `tqdm` alias pointing at
  `RichProgressBar`. The comment described the internal alias; the external package went unused when
  `rich` replaced it, and that replacement landed in one manifest only.
- **`scripts/check_manifests.py` makes the next one fail loudly**, in both directions: a third-party
  module the package imports **at module level** with no declaration in `pyproject.toml`, and a
  runtime dependency in `pyproject.toml` absent from `environment.yml`. Controlled against
  `origin/main`, where it reports the six missing runtime dependencies; clean on this tree.
- **The gate is module-level imports only, and the first version of this check got that wrong.** It
  walked every import and reported `optax`, `ot`, `pyvista`, `gmsh`, `cupy` and `colorlog` — all
  reached lazily inside functions or `try` blocks, each behind a guard, each degrading to a
  fallback. Gating there would demand declarations for backends the package deliberately treats as
  optional. `pyyaml` in #1687 was a module-level import, which is the failure that actually breaks an
  install. Lazy imports are reported, never gated.
- What this deliberately does **not** decide: whether an unused declaration should be removed.
  Absence of an import is not absence of use — `line-profiler` and `memory-profiler` are invoked as
  `kernprof` and `mprof` and are correctly declared without ever being imported. That direction
  needs a human.
