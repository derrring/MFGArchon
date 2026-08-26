- **`ghost_cell_fp_no_flux`'s vertex-centred branch used the cell-centred wall geometry**
  (Issue #2068). It computed `rho_i·(D + v_n·dx)/(D − v_n·dx)`, which is consistent only with a wall
  midway between ghost and interior at separation `2·dx`. On a vertex grid the wall **is** the
  interior node, so the branch satisfied its own stencil to machine zero while leaving a total flux
  that did not converge under refinement — it settled at `−v_x·rho` rather than going to zero.
  Corrected to `rho_i·(D + v_n·dx)/D`.

  **BREAKING for `ghost_cell_fp_no_flux(..., grid_type=VERTEX_CENTERED)`** and for
  `ZeroFluxCalculator` on a vertex grid. The path is public and exported and has no consumer in this
  repository, so nothing here breaks.

  It also removes a pole. The retired form is singular at `dx = D/v_n` — half the cell-centred limit,
  because the wrong geometry doubles the effective step — and returns a *negative density* past it.
  The corrected form is linear in `dx`; it still turns negative under strong inward drift beyond
  `dx > D/|v_n|`, which is a resolution bound rather than a blow-up.

  **The correction does not equalise the two centrings.** Against the exact profile
  `rho(s) = rho(wall)·exp(v_n·s/D)`, the vertex arm is the truncated series `1 + z` and the cell arm
  the Padé of `exp`, so the vertex wall imposes the condition to first order where the cell wall
  imposes it to second. A second-order vertex ghost needs two interior values and this signature
  carries one — a property of the interface, not something fixed here. The vertex half of the
  convention belongs to **#1904** and **#1935**, both open.
