The finite-difference ghost now mirrors about the wall node, because on a `TensorProductGrid` the wall is a node (#1935).

It used to copy the wall node (`u_g = u_0 + h g`), the cell-centred mirror. The 3-point Laplacian's wall row then returned `u''/2` at every h: on `u = cos(pi x)` the wall error was frozen at 4.94 (order 0.00), and 2-D edges and corners sat at 3/4 and 1/2 of the true value.

The ghost is now `u_g = u_m + 2(k+1) h (g - alpha u_b)/beta`, with u_m the node mirrored across the wall and u_b the wall node. One formula (`_write_wall_ghosts`) serves no-flux, reflecting, Neumann (alpha = 0, beta = 1) and Robin, in both ghost paths:

- the wall row converges at order 2.00, Neumann and Robin alike;
- `LaplacianOperator`'s matvec now agrees with its sparse assembly, which was node-centred already, to 1e-11.

Dirichlet, and Robin with beta = 0, keep their ghost.

**What moves.** Every consumer of the ghost: the FDM gradient, Laplacian, divergence and advection, so HJB-FDM's residual and its coupled solve. The regression golden moved at the wall: max|dU| 3.8e-04 at the wall node, 1.9e-04 inside. The semi-Lagrangian pair and particle FP read U's slope at the wall rather than the ghost since #2467 and #2470, and do not move.

**Not uniformly better.** HJB-FDM stays first order, the order of its upwind gradient. On four exact zero-flux stationary solutions its error changed by factors 0.74 to 32: three better, one 1.35x worse.

**Not changed.** `NeumannCalculator` / `RobinCalculator` (the `create_ghost_buffer_from_bc` path, no production caller) stay cell-centred and now disagree with the live ghost; a labelled test records that. The `grid_type` a `GhostCellConfig` carries is still not read. Both are #1919's.
