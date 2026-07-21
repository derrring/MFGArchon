- **A diverged HJB now stops the coupled loop before Fokker-Planck consumes it** (Issue #1717).
  `FixedPointIterator` solved FP with a NaN value function, so the FP solver failed first and an
  HJB divergence surfaced as an FP CFL diagnostic; that exception also escaped before the Issue
  #1078 HJB-vs-FP attribution could run. `solve()` now returns a `SolverResult` with
  `converged=False` and `convergence_reason="diverged_nan"` on this path rather than propagating
  the FP solver's `ValueError`, matching the two pre-existing divergence breaks in the same loop.
  `mass_conservation_error` is `None` when no FP step completed in the solve (the carried-over
  cold start has constant mass by construction and would measure as exactly 0.0), and remains a
  real number when an FP step did complete before the divergence.
