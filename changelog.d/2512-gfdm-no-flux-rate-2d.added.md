- **HJB-GFDM has rate oracles on reflecting walls in 2-D, one for each distinct discrete equation it ships** (#2512 (a)
  item 1; user ruling 2026-10-22).
  - The exact solution is exp17's reflected one, on Sobol clouds refined N_int 900, 1600, 2500 with the radius scaled
    with the spacing, on three scrambles.
  - `monotonicity_scheme="none"`: second order with Newton (EOC in [1.7, 2.3]) and first order with Howard (EOC in
    [0.7, 1.3]).
  - `"joint_socp"`: the same two bands, in the slow tier.
  - `HJBGFDMSolver`'s docstring now documents `inner_solver`, and states that in d >= 2 `"howard"` also changes the
    advection to a first-order upwind stencil.
  - `scripts/oracle_pins.txt` counts HJB-GFDM: 15 cells, 53 in all.
