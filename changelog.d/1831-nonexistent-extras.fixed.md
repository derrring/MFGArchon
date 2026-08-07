- **Six CI jobs stop asking for extras that do not exist** (Issue #1831). `pip install -e
  .[performance]` and `.[test,...]` name extras `pyproject.toml` never declared. pip does not treat
  that as an error — measured: `WARNING: mfgarchon 0.22.0.dev0 does not provide the extra
  'performance'`, **exit code 0** — so five jobs in `ci.yml` / `modern_quality.yml` and one in
  `security.yml` ran on base dependencies while their install line said otherwise, and stayed green
  the whole time. The references are deleted rather than the extras declared, because measured
  against what those jobs actually execute nothing was missing: `psutil` is a base dependency, not a
  performance extra, and none of the six imports numba, polars, joblib or either profiler, nor runs
  pytest — all six drive `python -c` scripts. So this changes nothing at runtime by construction; it
  removes a claim that was false. `security.yml` keeps `[dev]`, which exists, and drops only the
  `test` half.
  Two more instances outside the workflows go with them: `scripts/setup_development.sh` installs
  `".[dev,test,interactive]"` on both its uv and pip branches, and `interactive` does not exist
  either — so the documented developer-onboarding path has been handing every new contributor a
  silent partial install. `uv` behaves like pip here, warning and exiting 0. A tree-wide sweep
  after this change finds no remaining install site naming an undeclared extra.

