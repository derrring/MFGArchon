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
- **Six version floors disagreed after the first commit — and three of the six it created.**
  `origin/main` had three (`igraph`, `psutil`, `pydantic`); adding `meshio`, `osqp` and `scikit-fem`
  to `environment.yml` at looser floors than `pyproject.toml` made three more, and the second commit
  tightened all six. Measured at each of the three trees: 3, 6, 0. Presenting all six as
  pre-existing findings would have been false, and it is evidence against the "no floor check"
  decision rather than for it — a single PR introduced three divergences and hand-reading is what
  caught them. The decision stands only because #2167 deletes the file. `environment.yml` looser at
  every one: `meshio>=5.0` against
  `>=5.3`, `osqp>=0.6` against `>=1.0`, `psutil>=5.9` against `>=7.2.2`, `scikit-fem>=8.0` against
  `>=9.0`, `pydantic>=2.0` against `>=2.12.5,<3.0`, `igraph>=0.10` against `>=0.10.0`. A conda
  environment could satisfy the file while failing what `pyproject.toml` requires. Aligned once, by
  hand — deliberately **not** given a check, because #2167 deletes the file and a floor checker would
  be machinery built for a rival that is scheduled to go.
- **The file is marked `[SUPERSEDED-ON-ARRIVAL]` with a forward pointer to #2167**, at the top where
  a list view and an editor tab show it, and it says not to add dependencies there. It is reconciled
  rather than left broken because `scripts/manage_environments.sh` builds from it and an environment
  built that way could not run the suite — the #2158 measurement. Not because the documentation names
  it: README, CONTRIBUTING, AGENTS.md and `docs/` mention it zero times and say `pip install -e .`.
  The header also carries the two-line conda recipe that replaces it, since the reason people
  reached for conda — swapping the BLAS implementation — does not need a second manifest.
- **The check was wired into nothing.** `grep -rn check_manifests` over the whole tree returned one
  line — this changelog. Not in `.pre-commit-config.yaml`, not in a workflow, not in
  `scripts/local_ci.sh`, no unit test. Every sibling instrument is in the gate's self-test loop *and*
  has a step of its own; this one had neither, so the PR's central claim — "make the disagreement
  fail" — was delivered by nothing. The loop's own comment says why it exists: *"check_doc_api and
  capability_matrix have had one since they were written and this gate never invoked either."* It is
  in both now, plus `tests/unit/test_check_manifests.py`.
- **Only UNGUARDED module-level imports are gated, and that is the whole check.** `pyyaml` broke a
  fresh install because it was imported bare; a module-level `try: import cvxpy / except ImportError`
  cannot, because the module sets a flag and carries on. Measured on this tree: `yaml`, `numpy` and
  `rich` have unguarded sites; `cvxpy`, `torch`, `networkx`, `colorlog`, `optax` and `ot` are guarded
  at every module-level site. Gating the second group is how a check acquires false findings that
  teach people to ignore it — and the earlier version did exactly that to three of them. Guarded and
  undeclared is now reported, never gated.
- **The verdict no longer moves with the environment.** `packages_distributions()` knows only what is
  installed, so an import it cannot map went to an advisory list that never set the exit code — and
  the modules it cannot map are precisely the undeclared, uninstalled ones. Under CI's own install
  (`--group dev`, no backends) six more lost their mapping: 9 of 23 third-party module-level imports
  were structurally invisible while the check printed green. An unmapped name is now checked under
  its own name, and `--self-test` re-runs the real comparison with the mapping emptied and fails if
  the answer differs.
- **Ten import-time AST shapes were invisible**: module-level `with`, `for`, `while`, `match`, a
  class body, and `except*` — `ast.TryStar` is not an `ast.Try`, so both its body and its handlers
  were lost. Two are already in this codebase (`variational_problem.py:43` under
  `contextlib.suppress(ImportError)`, nine class-body imports). Twenty-four parametrised cases cover
  the shapes in both directions.
- **A file that does not parse is refused, not skipped.** A silent `continue` makes the check
  quietest about the files most likely to be wrong; `check_single_source.py` raises on the same
  condition and #1629 records the ruling.
- **`UNRESOLVABLE` was dead code with a self-test that tested something else.** Deleting the entry
  produced byte-identical output, and the self-test asked whether a distribution literally named
  `pytorch` was *installed here* — unrelated to whether the exemption is needed. It stayed green when
  the exemption was made dead, and when it was deleted; and with the 1.0.2 placeholder installed —
  the hazard it exists for — it advised deleting the guard. Split into `CONDA_TO_PYPI` and
  `IMPORT_TO_DISTRIBUTION`, each entry asserted still necessary against the manifests rather than
  against the environment.
- The header's BLAS recipe named `libblas=*=*mkl`, which **cannot solve on this platform**:
  conda-forge builds libblas against accelerate, newaccelerate, openblas, blis and netlib on
  osx-arm64, and mkl is x86-only. The recipe now names a variant that exists here, carries
  `-c conda-forge`, and says which platforms have mkl. Verified by dry-run.
- The PR body's before-count was wrong: `origin/main` is 31 conda + 4 pip, not 32 + 3. The fourth pip
  entry was `texttable`, which this change removes.
- **Two PRs each correct alone were wrong together.** The header's recipe read
  `uv pip install -e ".[all,dev]"`, and #2171 turns `dev` from an extra into a PEP 735 group.
  Measured on the merged trees: `WARNING: mfgarchon 0.22.0.dev0 does not provide the extra 'dev'`,
  **exit 0** — the #1658 shape, in the recipe written to justify retiring this file. Neither PR's
  guard could see it, because `environment.yml` was outside the scanned population. The recipe now
  reads `-e ".[all]" --group dev`, and #2171's scan includes this file. This ordering is real:
  #2171 must merge first, or the recipe names a group that does not yet exist.
