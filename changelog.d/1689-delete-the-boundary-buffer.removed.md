- **`PrecomputedMonotoneStencils`, `joint_socp`'s boundary buffer, is deleted** (#1689 PR 2, sites a, b and h, member 1).
  - It rewrote the Laplacian rows of HJB-GFDM's boundary nodes with M-matrix QP weights, taking its QP's unconstrained
    weights on a failed solver status. Every boundary row is replaced by its BC row (Dirichlet, Neumann, no-flux, or
    Robin with alpha = 0 and beta = 1) before the Newton or Howard solve, except a Neumann node's row under
    `use_ghost_nodes=True`, which keeps its PDE row. Measured with the buffer switched off against on, on 1-D and 2-D clouds, mixed Dirichlet
    and Neumann walls, an obstacle with the visibility filter and local coordinate rotation, under the Newton and the
    Howard inner solvers: the solves were identical, bit for bit.
  - `HJBGFDMSolver(monotonicity_scheme="joint_socp", use_ghost_nodes=True)` is refused by name. Ghost nodes keep the
    PDE row at a Neumann boundary node, and `joint_socp` gives monotone weights to interior rows only. The buffer raised
    `IndexError` on most such clouds. On a 1-D cloud whose only Neumann node was index 0, whose ghost indices alias the
    cloud's last points, it solved with the buffer's weights in that row, and switching the buffer off moved U by
    2.1e-01.
  - Under `joint_socp`, `assemble_hjb_iteration_matrix` and `compute_all_derivatives` now return the Wendland-Taylor
    Laplacian weights at boundary nodes, where they returned the buffer's. No solve reads those rows.
- **`osqp` is no longer a default dependency**: nothing in the library uses it after #1689's boundary buffer was
  removed. `cvxpy`, which `joint_socp` needs, still installs it as one of its own solvers.
