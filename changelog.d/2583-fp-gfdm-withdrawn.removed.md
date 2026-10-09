- **BREAKING. `FPGFDMSolver` is withdrawn: construction raises `NotImplementedError` with the reason and an
  alternative** (#2583). FP-GFDM built no wall. Its operator, a `TaylorOperator` over scattered collocation
  points, took no boundary-condition argument and no flux or ghost row was ever built, so a declared NO_FLUX
  wall was accepted and read by nothing and the drift's flux crossed the boundary. What that did to mass
  depended on the drift. Measured on 1-D, 21 points, sigma = 0.4, T = 0.5, Nt = 10, uniform initial density
  and an explicit no-flux wall:
  - under a drift +1 toward `x_max` (FP-FDM given `potential_field = -x`, FP-GFDM `drift_field = +1`), FP-FDM
    piles the density up at the wall (m(T) = 0.0493, 0.5573, 6.3400 at x = 0, 0.5, 1), while FP-GFDM stayed
    uniform (1.0000 at all three), both at mass 1.0000;
  - under the drifts `x` and `0.5 - x`, FP-GFDM's mass at T was 0.599 and 1.629.

  An implicit domain carrying no BC gave the same uniform solve, and every domain GFDM is admitted on is
  bounded. FP-GFDM was reachable as a standalone FP solve with a velocity the caller supplies,
  `solve_fp_system(drift_field=...)`. A coupled solve with the GFDM pair already raised #1420's auto-route
  error.
- **Three routes raise, with the same message:** `FPGFDMSolver(problem, ...)`,
  `create_paired_solvers(problem, NumericalScheme.GFDM, ...)` and `problem.solve(scheme=NumericalScheme.GFDM)`.
  The pair now refuses before it builds the GFDM HJB. `problem.solve(scheme=GFDM)` used to raise
  `HJBGFDMSolver`'s `TypeError` for the missing `collocation_points` instead, and it reaches the reason now.
- **What to use instead, on a grid:** a scheme with a dual FP, e.g.
  `problem.solve(scheme=NumericalScheme.FDM_UPWIND)`. To keep the GFDM HJB, pair it with an FP solver that
  builds walls: `problem.solve(hjb_solver=HJBGFDMSolver(problem, collocation_points=...),
  fp_solver=FPFDMSolver(problem))`. That pair is not dual, and Expert Mode warns about it.
- **On an implicit domain, neither of those runs:** `FDM_UPWIND` and `FPFDMSolver` refuse the domain. The
  rebuild, as the discrete adjoint of the GFDM generator that the HJB side already assembles, is #2584.
- **The class stays, as a stub,** so `from mfgarchon.alg.numerical.fp_solvers import FPGFDMSolver` still
  imports and fails with the reason rather than with an `ImportError`.
  - The strong-form solve body is deleted.
  - With it go FP-GFDM's mass gate (the clip gate and the mass-drift warning added for #1683 and #1752) and
    their tests.
  - #1752 (the unstabilised central flux diverges under refinement) describes that body, so it closes as
    superseded by #2584.
  - The duality warning no longer recommends the GFDM pair.
- **No deprecation window, by ruling.** A window would keep a solve that returns a wall-less density on every
  domain it accepts. The user ruled on 2026-10-09 to refuse it now.
