- **`PrecomputedMonotoneStencils`, `joint_socp`'s boundary buffer, is deleted** (#1689 PR 2, sites a, b and h, member 1).
  - It rewrote the Laplacian rows of HJB-GFDM's boundary nodes with M-matrix QP weights, taking its QP's unconstrained
    weights on a failed solver status. No solve read those rows: every boundary row is replaced by its BC row (Dirichlet,
    Neumann or no-flux) before the Newton or Howard solve. Measured with the buffer switched off against on, on 1-D and
    2-D clouds, mixed Dirichlet and Neumann walls, an obstacle with the visibility filter and local coordinate
    rotation, under the Newton and the Howard inner solvers: the solves were identical, bit for bit.
  - `HJBGFDMSolver(monotonicity_scheme="joint_socp", use_ghost_nodes=True)` is refused by name. Ghost nodes keep the
    PDE row at a Neumann boundary node, and `joint_socp` gives monotone weights to interior rows only. It raised
    `IndexError` inside the buffer before.
- **`osqp` is no longer a default dependency**: nothing in the library uses it after #1689's boundary buffer was
  removed. `cvxpy`, which `joint_socp` needs, still installs it as one of its own solvers.
