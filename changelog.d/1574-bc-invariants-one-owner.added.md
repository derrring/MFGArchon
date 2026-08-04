- **`mfgarchon/geometry/boundary/invariants.py` — one owner for what a BC asserts about a solved
  field** (Issue #1574). `seam`, `mass_drift` and `bc_residual`, plus `RESIDUAL_IS_EXACT`, which
  records per BC type whether its residual is **zero in exact arithmetic** (absolute tolerance is
  valid: a periodic seam, a Dirichlet wall value) or **only in the limit** (only a convergence
  trend may be asserted: mass, a normal derivative). Confusing those two is the recurring error
  this file exists to stop -- asserting `mass drift < 1e-9` reported six solvers as defective when
  the drift halves per refinement.
  It lives in the library, not in `tests/`, because tests are not the only consumer:
  `scripts/capability_matrix.py` drives the same surface, and #1574's phase 1b wants the
  declaration gate itself to reach these. The seam was previously reimplemented inline in three
  test files; those copies are gone.
- `bc_residual` **raises** for a BC type with no residual defined rather than returning 0.0, since
  returning zero would certify the type as satisfied — the exact failure the module exists to make
  impossible.
