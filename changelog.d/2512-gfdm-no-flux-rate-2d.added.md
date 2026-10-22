- **HJB-GFDM's default scheme has a rate oracle on reflecting walls in 2-D** (#2512 (a) item 1).
  - `monotonicity_scheme="none"` with Newton converges at second order to exp17's exact reflected solution on Sobol
    clouds. The pin is EOC in [1.7, 2.3] over N_int 900, 1600 and 2500, with the radius scaled with the spacing, on
    three scrambles.
  - `HJBGFDMSolver`'s docstring now documents `inner_solver`, and states that in d >= 2 `"howard"` also changes the
    advection to a first-order upwind stencil.
