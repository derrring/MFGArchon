- **The measure on a node-centred grid is the trapezoid, and the wall node owns half a cell**
  (Issue #2145, closing #1904/#1935). `TensorProductGrid` builds `linspace(lo, hi, N)`, so the wall
  lies ON `x_0` and the two end nodes own `h/2` each. Every conservative FP path divided its wall
  flux divergence by a full cell instead, which made the schemes telescope against rectangle weights
  — so `sum(m)*dx` was conserved to machine precision while the real mass was not. Measured on the
  #1975 census fixture (sigma 0.3, drift 3.2, 81 nodes, 200 steps), rectangle / trapezoid drift:
  `divergence_upwind` went from `-0.00000% / -19.26153%` to `+25.37733% / -1.98e-12%`, and
  `divergence_centered` from `-0.00000% / -25.45389%` to `+37.00513% / -2.44e-12%`. A quarter of the
  density was leaving under tests reporting machine zero.

  Four sites carried the same error and are fixed together, because they cooperate: the wall
  assembly in `fp_fdm_bc`, the separate one in `fp_fdm_alg_divergence_centered`, the finite-volume
  advection kernel `fp_fvm_flux.axis_flux_divergence`, and `AdvectionOperator`'s no-flux cells.
  Fixing a subset is worse than fixing none — with diffusion on `h/2` and advection on `h` the
  explicit step conserved neither measure (rectangle `1.00000000 -> 0.40908779`).

- **`LaplacianOperator`'s conservation/accuracy trade-off did not exist** (Issue #2145). Its
  `mass_conservative` flag documented a choice between a second-order wall that "leaks mass
  (`1ᵀL ≠ 0`)" and a first-order wall that conserves. Both branches are exactly conservative, for
  different measures: measured on n = 5/9/21, the first has `max|1ᵀL| = 16/64/400` and
  `max|wᵀL| = 0`, the second the reverse. `1ᵀL = 0` is column conservation under uniform weights,
  i.e. of `sum(m)`, which is not the mass here. With the wall rows on the correct control volume the
  two branches are byte-identical and both satisfy `wᵀL = 0`, so the documented cost of first-order
  wall accuracy was the price of the wrong measure.

- **`FPFVMSolver` is vertex-centred and second order** (Issue #2145). It interpreted the grid's
  nodes as cell centres while placing them on the walls, so its `N` cells tiled a domain of length
  `L + dx`. Read back from the uniform equilibrium of a no-flux diffusion solve, the effective
  domain was exactly `1 + h` at every resolution (`1.050000 / 1.025000 / 1.012500 / 1.006250`); in
  `d` dimensions the inflation compounds as `(1 + h/L)^d`. Against the analytic
  `1 + 0.5 cos(pi x) exp(-D pi^2 t)` with `dt ~ dx^2`, MUSCL converged at order 0.91/0.96 with the
  error maximal at the wall (`8.85e-03` against the FDM's `8.67e-05`) and decaying inward. It now
  takes its control volumes from the grid, reads an effective domain of `1.000000`, converges at
  2.00/2.00, and is byte-identical to the divergence-form FDM on pure diffusion — which its own
  docstring always said it should be.

- **`MFGProblem.spatial_shape` is owned by the geometry** (Issue #1888 follow-up). Three branches
  recomputed it; the `d >= 4` one read `num_spatial_points`, a count rather than a shape, so
  `spatial_bounds=` built a flat `(14641,)` density where `geometry=` built `(11, 11, 11, 11)` for
  the same grid. Nothing compared a field against the grid shape until `geometry.integrate` did.
