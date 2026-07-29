The inner HJB Newton now tests convergence on the **residual** rather than on the step norm
(#1745). It previously declared `converged` when `norm(delta_U) * sqrt(dx)` fell below the
tolerance, which a stalled iteration or a large Jacobian produces without the iterate being a
root. Measured on 1-D FDM_UPWIND solves, the step criterion accepted iterates whose HJB residual
was 59-77x the requested tolerance at N=21 and 267-402x at N=41 -- it got worse under
refinement. A non-converged inner solve is now reported instead of being swallowed by an empty
branch: `sigma=0.05` on a 21-point grid turns out never to have solved, and now says so 40 times
with residuals up to 4.9e+05 where it previously returned silently. The golden baselines were
regenerated; the new answer's final residual is 95x smaller at that configuration.
