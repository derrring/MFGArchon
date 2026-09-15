- **The 1-D FDM HJB inner Newton no longer stops at its first overshoot, and a coupled result no longer
  reports convergence over inner solves that are not roots** (Issue #1878).
  - **The guard.** A non-decrease guard in `solve_hjb_timestep_newton` stopped Newton as soon as a step
    raised the residual. On this monotone system the first step routinely does, and the iterates converge
    after it. The guard returned an iterate worse than its start. On the 1-D smoke fixture at 6c0610d2,
    Picard reported `converged=True` at sweep 38 while three of ten backward steps had returned iterates
    with residuals up to 8.7e+02. That U was 2.57 away from the scheme's own solution.
  - **Removed.** The guard is gone, and the same fixture converges at sweep 31 with every backward step a root.
  - **Budget.** `max_newton_iterations` is the number of steps taken. A solve that spends it reports the
    residual of the iterate it returns, and so does a budget of 0. The warning counts the steps actually
    taken; it used to print the budget.
  - **Uncomputable steps.** A step that cannot be computed (non-finite residual or Jacobian, failed linear
    solve) stops the solve instead of re-measuring one point until the budget runs out.
  - **The report.** `HJBFDMSolver.inner_solve_failures()` lists the latest sweep's non-converged time steps,
    on the 1-D and nD paths. `FixedPointIterator` reports `converged=False`, with reason
    `inner_hjb_not_converged: ...` and `metadata["inner_hjb_failures"]`, when the sweep that met its criteria
    had any. A duck-typed HJB solver without the method is read as untracked.
  - **Visible behaviour changes.**
    - A solve that cannot converge now spends its whole budget: the review measured up to 9x the guarded
      time on non-convex Hamiltonians.
    - A result that used to say `converged=True` over non-roots can now say `converged=False`. The smoke
      fixture refined to 81 nodes needs 32 Newton steps at t_idx 9 against the default budget of 30, so it
      reports `converged=False` at sweep 33. With `max_newton_iterations=60` it converges.
  - **Not covered.** Other coupling loops do not read `inner_solve_failures()` yet.
