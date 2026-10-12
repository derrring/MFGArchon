- **n-D HJB-FDM's cached Laplacian reads a time-dependent boundary value at the step's time** (#2543's time half;
  #2512, row B2).
  - The operator is built once and kept the clock it was built with, t = 0. The gradient reads the step's time, so
    the two halves of the residual read the wall at different times.
  - A Neumann wall, held by the ghosts alone since #2537, therefore converged to the t = 0 datum.
  - Measured on #2543's fixture: 2-D, `neumann_bc` with value -0.5 - 4t on every face, sigma = 0.5, T = 0.4,
    Nt = 4. The mean outward slope at x_min of the t = 0.3 slice was, at 21², 41² and 81²:
    - at 52219be2: -0.983, -0.806, -0.675, tending to g(0) = -0.50;
    - now: -1.680, -1.695, -1.699, tending to g(0.3) = -1.70.
  - A BC object replaced after the operator was built is still solved stale (#2475's class, not this fix).
