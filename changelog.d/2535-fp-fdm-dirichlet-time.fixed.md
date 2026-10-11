- **FP-FDM reads a Dirichlet value at the level it solves for and at the wall node, and no longer overwrites a
  no-flux wall** (#2535; #2512, row B2). This is FP-FDM's own BC channel, `FPFDMSolver(problem,
  boundary_conditions=bc)`. On the shared route, `fp_view_of_shared_bc` still makes a Dirichlet wall absorbing,
  m = 0, by design.
  - **A callable Dirichlet value is honoured.** Level k + 1 reads it at t_{k+1}, at the node, in every dimension.
    In 1-D, level 0 reads it at t = 0.
    - Measured with the exact solution m = 1 + 0.5 (|x|² + 2dDt) of the zero-drift equation (σ = 0.6, T = 0.5,
      Nt = 10, walls held at m), at 76d8b1dc:
      - in 1-D with a per-face `g(t)`, the error at T was 9.000e-2, which is 2dDbT: the walls were held at g(0);
      - in 2-D with a uniform `g(t, x)`, the error was 2.18: the n-D path read a uniform BC's callable as 0.
    - Both are now below 2e-15.
  - **The 1-D post-step overwrite is gone.** After every step it wrote the BC's value onto both walls whenever all
    segments were Dirichlet. That included a wall left to a no-flux `default_bc`, which read the default value.
    - Measured at 76d8b1dc: one Dirichlet segment, 2.0 on x_min, under a no-flux default held x_max at 0.0 at
      every level, from an initial density of 1.0.
    - It now stays at or above 1.0, as mass flows in from the wall held at 2.
    - The implicit step's Dirichlet rows hold g(t_{k+1}) without the overwrite, and the oracle above is exact
      without it. The explicit steps refuse a Dirichlet wall.
    - Level 0 is written as before, only for a BC whose segments are all Dirichlet, but now only on its Dirichlet
      walls. A mixed BC keeps the caller's initial density there.
  - **The n-D Dirichlet read has no fallbacks.** It returned 0.0:
    - for any callable value on a uniform BC;
    - on an `AttributeError`, `ValueError` or `IndexError`;
    - for a node on no Dirichlet wall.

    It now reads through `get_bc_value_at_boundary` or raises.
