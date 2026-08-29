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
- **One job ran `--group` without upgrading pip first, and the test written to prevent exactly that
  passed vacuously.** `security.yml::license-compliance` is a separate job — no `needs`, no cache, its
  own runner — so the upgrades in the other two jobs of that file never reached it. The check asserted
  `"install --upgrade pip" in text` over the whole file, and those other jobs put the string there: a
  file-scoped assertion for a job-scoped property. It now walks the YAML per job and fails naming the
  job. Independently measured before and after: eleven `--group` sites, ten preceded by an upgrade,
  one not; zero after.
- **The guard was blind to three real install spellings**, found by planting them: `pip install
  ".[x]"` with no `-e`, a bare `pip install .[x]`, and a backslash-continued `-e \` / `".[x]"`. It
  also did not know `uv`'s flag form at all — `uv run --extra dev` is a **hard error** once `dev` is a
  group, where pip's bracket form only warns and exits zero. Continuations are now joined before
  matching, the subject may be any local-path spelling, and `--extra` / `--no-group` are read.
- Four live `uv run --extra dev` sites would have hard-errored after the move: `AGENTS.md`,
  `tests/conftest.py` ×2, `tests/unit/test_mfg_caplog.py`. That is a fifteenth call-site spelling the
  original enumeration did not have, and it is outside the guard's scanned population as well —
  doubly invisible until the extractor learned the flag.
- **A bracket now only counts inside an install command.** Widening the subject to a bare `.` made the
  scan read `[0-9]*.[0-9]*` in a comment about version matching as the extra `0-9`, and prose *about*
  the #1658 incident as an instance of it. Comment lines are deliberately **not** skipped: `pyproject.toml`
  documents its own install commands in comments, and a documented command naming a nonexistent extra
  is the #2170 class exactly.
- `--group <[path:]group>` is pip's documented syntax; the group name is the part after the colon.
  Before, `--group pyproject.toml:dev` failed the guard by reading `pyproject.toml` as the name.
- **The sentinel protected one of its four sources.** `assert count > 20` could not fail while
  `mfgarchon/**/*.py` supplied hundreds, so dropping `scripts`, `docs` or `Makefile` from the
  population was silent — measured, all three passed. It now asserts per source, with the expected
  roots **written out rather than read from the constants they check**: iterating `ROOTS` meant
  deleting an entry also deleted its own check. Four population mutations killed.
- `numerical` no longer carries a `# Development tools (not user-facing)` header — it is in
  `[project.optional-dependencies]` precisely because it is user-facing. The header belonged to `dev`,
  which moved.
- **`environment.yml` joins the scanned population.** It carries install commands in its header, and
  being outside the scan is what let a cross-PR defect through: #2166's recipe reads
  `-e ".[all,dev]"`, which this change makes an extra that does not exist — `WARNING: … does not
  provide the extra 'dev'`, exit 0. Each PR was green alone and nothing checked the pair. Verified by
  real `git merge` of all five branches, not by overlaying archives: an archive overlay silently
  restores the second branch's copy of files the first one changed, and gave the wrong answer twice
  before that was noticed.
