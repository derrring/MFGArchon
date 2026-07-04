- **Python 3.13 and 3.14 support, verified in CI.** New additive `python-compat.yml` workflow runs the
  install + fast test suite across 3.12/3.13/3.14 weekly and on any change to the matrix or dependency
  metadata (kept off the per-PR and nightly paths, so no extra cost there). Added per-version classifiers
  (3.12/3.13/3.14); `requires-python` stays `>=3.12` (no upper cap). Python 3.15 (pre-release, ships
  Oct 2026) runs as an allow-failure early-warning job, not a support claim.
