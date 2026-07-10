- **FP drift coefficient fails loud on a smooth non-quadratic-MINIMIZE Hamiltonian** (Issue #1542,
  RFC #1574 Phase 0). `fp_drift_coefficient` silently fell back to `coupling_coefficient` (default 0.5)
  for a MAXIMIZE-quadratic or Moreau-Yosida-regularized `SeparableHamiltonian` — regimes the drift
  router steers to the scalar `-c*grad(U)` path (it gates on `is_smooth()` alone), where that form is
  the wrong optimal control (wrong sign for MAXIMIZE, wrong form for a regularized cost). It now raises
  `NotImplementedError` naming the case instead of advecting with silently-wrong physics. The
  quadratic-MINIMIZE path (every stock/published config) is unchanged; only the previously silent-wrong
  regimes are affected.
