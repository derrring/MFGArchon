- **`uv.lock` is tracked again, generated from `pyproject.toml`** (Issue #2167, step 2). It was
  retired in #2138 for being five months stale, unsatisfied by `pyproject.toml` and read by nothing;
  this one is current, satisfied, and the artifact the project's dependency set resolves to.
- **No `[tool.uv] environments` restriction.** Bounding the lock to darwin was considered and
  rejected on measurement: **all 27 CI jobs run `ubuntu-latest`**, and `uv sync` on a platform the
  lock does not declare is a hard error — `The current Python platform is not compatible with the
  lockfile's supported environments`. Restricting is exclusion, not deferral. The cost of covering
  everything is small on this project, unlike on the single-package probe that first suggested
  otherwise: universal is 153 packages / 601 KiB, darwin+linux 151 / 492 KiB, darwin-only 132 /
  198 KiB, and resolution takes seconds in every case. (Re-measured on this branch after review;
  the first table had each restricted arm one package high, from a slightly different
  `pyproject.toml`.) The 15 `nvidia-*` entries are marker-gated to linux and are not
  downloaded elsewhere.
- The guard added in #2169 — a tracked lock must not record a ruff disagreeing with the pin — passes,
  and is now unreachable by construction: ruff left the dev group in #2172, so no resolver sees it.
  Its docstring says so, because an unreachable guard read as dead invites deletion.
