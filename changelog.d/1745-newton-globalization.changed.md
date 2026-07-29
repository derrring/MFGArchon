`NewtonSolver` no longer accepts a step that increases the residual (#1745). `line_search`
defaults to `True`, an always-on divergence guard stops the solve when the residual exceeds
1e4x its initial value (PETSc's `SNESSetDivergenceTolerance` default), and "no acceptable step"
and "diverged" are now named outcomes in `SolverInfo.extra` rather than an exhausted iteration
budget. The Armijo condition was also comparing a residual norm against a step norm; it now uses
the standard `||F(x + a*d)|| <= (1 - c*a)*||F(x)||`. Measured on a 2-D HJB solve, the previous
default converged to 1.57e-03 in five iterations and then diverged geometrically to 2.42e-01
because the full step increased the residual where a half step would have quartered it.
