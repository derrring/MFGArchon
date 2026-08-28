- **Development tooling moved out of the published extras into PEP 735 `[dependency-groups]`**
  (Issue #2167, step 1). `dev` and `docs` are not part of the installable interface: nobody installs
  this package in order to get its linter. A dependency group is read from the source tree and never
  reaches the built distribution's metadata, so it cannot be requested by a downstream install —
  which is right for tooling and wrong for a backend. Compute backends (`nn`, `all`, `numerical`,
  `core`) stay in `[project.optional-dependencies]`, because a user must be able to ask for them by
  name from a published wheel. Getting the two backwards makes the backends uninstallable.
- **Fourteen call sites changed**, from `pip install -e .[dev]` to `pip install -e . --group dev`:
  `ci.yml` ×3, `security.yml` ×3, `nightly.yml` ×2, `deprecation-check.yml`, `discrimination.yml`,
  `python-compat.yml`, `Makefile`, `setup_development.sh` ×2. Verified end to end against pip 26.2.1
  before any of them were touched: `--group` composes with `-e .`, resolves `include-group`
  transitively, can be combined with an extra in one command, and errors on a name that does not
  exist. Every workflow already runs `python -m pip install --upgrade pip` immediately before its
  install, so the pip ≥ 25.1 floor for `--group` is met — and a test now asserts that pairing rather
  than leaving it as a coincidence.
- **Three of those fourteen were found by the new guard, not by the search that preceded it.**
  `deprecation-check.yml`, `discrimination.yml` and `nightly.yml` spell it `-e ".[dev,numerical]"`,
  and the pattern used to enumerate the call sites matched only the single-extra form. A population
  predicate that misses a spelling reports a clean sweep.
- **A guard against the whole class**: `tests/unit/test_install_commands_name_real_extras_2167.py`
  fails when any install command in the repository names an extra or group that `pyproject.toml`
  does not declare. This has shipped here before — `nightly.yml` referenced a `numerical` extra for
  three and a half months while none existed, and because pip installs an unknown extra with a
  warning and a zero exit, the SOCP gate failed on `ImportError` every night and burned one of ten
  `--maxfail` slots (#1658). Nothing has guarded it since.
- The guard's own population is the thing that can be wrong, so `test_the_scan_finds_the_known_call_sites`
  is a sentinel: a glob that stops selecting files reports zero violations, which reads exactly like
  a clean repository. Extension-free files are scanned deliberately — `Makefile` has no suffix, and a
  suffix allowlist is how a scan silently loses one.
- **It found eleven pre-existing sites where the library tells a user to install an extra that does
  not exist** — `pip install mfgarchon[neural]`, `[performance]`, `[reinforcement]`,
  `[visualization]`, `[gpu]`, `[optimization]`, `[jax]`, `[geometry]`. pip warns and exits zero, so
  the user follows the advice, sees success, and still cannot import torch. Recorded in #2170 and
  pinned as an **exact** set keyed by `(file, extra)`: a new one fails immediately, and fixing one
  requires deleting its entry, because a floor would rot into a list of things fixed years ago with
  nothing to say so. Keyed by file rather than line, since a line number in a durable artifact
  expires on the next edit above it.
