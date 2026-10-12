- **HJB-GFDM's Newton step imposes its boundary values, and returns a root of the system it solves** (#2599). The step
  tested convergence, and measured its Armijo line search, on the PDE residual alone; the system it solves replaces
  each boundary row with the BC's, u - g at a Dirichlet point. One owner now returns the residual with its boundary
  rows, and the convergence test, the merit and each trial's norm all read it.
  - **A spatially constant iterate with no source has a zero PDE residual.** The step stopped at iteration 0 and
    returned the terminal unchanged, imposing no boundary value. Measured at 52219be2 on a 1-D fixture with walls at
    0.7 and 0.4: a terminal of 0 or 0.1 left both walls at 0 or 0.1. In 2-D (6 x 6, g = x + 2y), Newton left the
    boundary 2.9 from g with a constant terminal; Howard imposed it. Both now hold g. A Neumann wall is affected
    the same way: with a value of 0.3 and a constant terminal, every step ended at a residual of 0.42 at 375122e5,
    and now ends below 1e-7.
  - **The merit was not the function the Newton direction descends, with any BC.** It read the PDE residual at
    every point, including the boundary points whose rows the step replaces with the BC's. The search backtracked
    to its floor, and steps ran out of iterations in silence. On the regression golden's fixture, a no-flux BC on
    which no trial's projection moved u, 290 of 300 iterations ended at the floor (375122e5). Every step ran all 30
    iterations and stopped between 6.3e-01 and 2.8e-04 against a tolerance of 1e-6. Each step now converges in 2
    or 3 iterations, to at most 9.4e-10. The golden was regenerated: max |dU| 2.353e-02.
  - **A step that runs out of iterations says so.** It logs a warning naming the time index, the residual with its
    boundary rows, and the tolerance. Steps that still do not converge include a zero terminal against walls of 0.7
    and 0.4, and the 2-D Newton cases above, as on 52219be2, where they were silent. #2610 records them: at the
    first such step of the zero terminal, the step's system has no root near any start tried.
  - The residual is still unscaled against `DEFAULT_NEWTON_TOLERANCE` (#1885).
