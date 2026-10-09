- **BREAKING. `FPGFDMSolver` is withdrawn: construction raises `NotImplementedError` naming the
  alternatives** (#2583). FP-GFDM built no wall. Its operator, a `TaylorOperator` over scattered collocation
  points, took no boundary argument, and no flux or ghost row was ever built, so a declared NO_FLUX wall was
  accepted and read by nothing. Measured on 1-D, 21 points, sigma = 0.4, T = 0.5, Nt = 10, uniform initial
  density, an explicit no-flux wall and a drift toward `x_max` (FP-FDM given `potential_field = -x`, FP-GFDM
  `drift_field = +1`): FP-FDM piles the density up at the wall (m(T) = 0.0493, 0.5573, 6.3400 at
  x = 0, 0.5, 1) while FP-GFDM stayed uniform (1.0000 at all three), both at mass 1.0000. A mass check cannot see it. An
  implicit domain carrying no BC gave the same, and every domain GFDM is admitted on is bounded, so no
  reachable problem had a correct FP-GFDM solve. **Two routes raise:** `FPGFDMSolver(problem, ...)` and
  `create_paired_solvers(problem, NumericalScheme.GFDM, ...)`, with the same message.
  `problem.solve(scheme=NumericalScheme.GFDM)` is unchanged: it already raised `TypeError`, because it has
  no way to pass the `collocation_points` that `HJBGFDMSolver` requires.
- **What to use instead.** On a bounded problem, use a scheme with a dual FP, e.g.
  `problem.solve(scheme=NumericalScheme.FDM_UPWIND)`. To keep the GFDM HJB, pair it with an FP solver that
  builds walls: `problem.solve(hjb_solver=HJBGFDMSolver(problem, collocation_points=...),
  fp_solver=FPFDMSolver(problem))`, or `FPParticleSolver`. That pair is not dual, and Expert Mode warns about
  it. The rebuild, as the discrete adjoint of the GFDM generator that the HJB side already assembles, is
  #2584.
- **The class stays, as a stub**, so `from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver` still
  imports, and the GFDM pair fails with the reason rather than with an `ImportError`. The strong-form solve
  body is deleted, and with it FP-GFDM's mass gate: the clip gate and the mass-drift warning added for
  #1683 and #1752, and their tests. #1752 (the unstabilised central flux diverges under refinement)
  describes that body, so it closes as superseded by #2584.
- **No deprecation window, by ruling.** A window would keep a solve that returns a wall-less density,
  conserving mass, for every reachable problem. The user ruled on 2026-10-09 to refuse it now.
