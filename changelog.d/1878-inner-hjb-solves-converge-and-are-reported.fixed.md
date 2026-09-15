- **The 1-D FDM HJB inner Newton no longer stops at its first overshoot, and a coupled result no longer
  reports convergence over inner solves that are not roots** (Issue #1878).
  - The non-decrease guard in `solve_hjb_timestep_newton` stopped Newton as soon as a step raised the
    residual. On this monotone system the first step routinely does, and the iterates converge after
    it. The guard returned an iterate worse than its start: on the 1-D smoke fixture at 6c0610d2,
    Picard reported `converged=True` at sweep 38 while three of the ten backward steps sat at residuals
    7.5e+01 to 2.8e+02 against 1e-06, and the reported U was 2.57 away from the scheme's solution.
  - The guard is removed. The same fixture converges at sweep 31 with every backward step a root.
  - A solve that spends its step budget now reports the residual of the iterate it returns.
  - The warning's count is the steps that iterate took. It used to print the budget.
  - `HJBFDMSolver.inner_solve_failures()` lists the latest sweep's non-converged time steps, on the 1-D
    and nD paths.
  - `FixedPointIterator` reports `converged=False`, with reason `inner_hjb_not_converged: ...` and
    `metadata["inner_hjb_failures"]`, when the sweep that met its criteria had any.
  - Other coupling loops do not read `inner_solve_failures()` yet.
