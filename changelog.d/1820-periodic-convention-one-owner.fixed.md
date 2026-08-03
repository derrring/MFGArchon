- **Two periodic-BC convention errors, in opposite directions, on the same grid** (Issue #1820).
  `TensorProductGrid` is endpoint-inclusive: `field[0]` and `field[-1]` are both real nodes and the
  same physical point, so the period is `L` and there are `N - 1` distinct degrees of freedom.
  - `enforce_periodic_value_nd` used the **halo** form (`field[0] = field[-2]`,
    `field[-1] = field[1]`), correct only for an array carrying a ghost cell per side. Applied to a
    halo-free grid it moved both endpoints one cell in opposite directions: on `sin(2 pi x)` with
    `N=21` it took a seam of `2.4e-16` to `6.2e-01`, after **every** semi-Lagrangian timestep. It is
    now the identification of the two nodes (their mean, privileging neither end).
  - `_crank_nicolson_periodic_1d` wrapped all `N` nodes, taking `U[N-1]` as node 0's left neighbour
    -- a neighbour at distance 0 rather than `dx`. Measured against the analytic periodic heat
    kernel `exp(-D k^2 t) sin(k x)`: max error `1.19e-01`, not shrinking under refinement. Solving
    on the `N - 1` distinct DOFs and restoring the duplicate gives `6.1e-04` and an exact seam.
  - `HJBSemiLagrangianSolver`'s periodic seam falls from **7.68e-01 to 2.45e-16**, and `FPSLSolver`
    and `FPSLAdjointSolver` from `1.25e-01` to exactly zero.
- **A regression this unmasks, stated rather than shipped quietly**: `FPSLSolver` mass drift over a
  periodic solve moves from `7.19%` to `10.77%`. Mass was already badly non-conserved and the seam
  error was partly cancelling it; closing the seam removed the cancellation. Mass is now a second,
  independent invariant in the #1822 ratchet, where six FP solvers fail it -- including
  `FPFVMSolver`, which satisfies the seam at `2.2e-16` while creating 6.5% of its own mass.
