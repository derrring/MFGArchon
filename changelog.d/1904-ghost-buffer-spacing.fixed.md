- **The ghost buffer no longer falls back to `dx = 1.0`** (Issue #1904, first of three steps).
  `PreallocatedGhostBuffer` derived its spacing only from `domain_bounds`, which almost no caller
  passes — `pad_array_with_ghosts(..., geometry=None)` left `_grid_spacing = None` and every
  consumer read `dx = 1.0`. An inhomogeneous Neumann condition was therefore applied as `g/h`
  instead of `g`, so the recovered `du/dn` **diverged as `1/h`**:

  | Nx | requested `du/dn` | recovered, before | after |
  |--:|--:|--:|--:|
  | 21 | 2.0 | 11.83 | **2.0000** |
  | 41 | 2.0 | 23.71 | **2.0000** |
  | 81 | 2.0 | 47.44 | **2.0000** |

  Robin read the same fallback, in the denominator `alpha + beta/dx`. It now satisfies its own
  condition at the wall: `alpha*u + beta*du/dn - g` measures **exactly 0** for `beta = 1` and
  converges at O(h) otherwise (`2.5e-03 → 6.3e-04 → 1.6e-04` over Nx 21/81/321).

- **The spacing was already in hand at every site that needed it.** `laplacian_with_bc` takes
  `spacings` as a parameter and dropped it at the padding call; `_compute_gradient_array_1d` has
  `Dx`. `pad_array_with_ghosts` and `PreallocatedGhostBuffer` now accept an explicit `spacing`
  (scalar or one per axis, wrong length raises rather than broadcasting), and the three call sites
  that hold it pass it. No-flux and `neumann(0)` are byte-identical, asserted — the flux term
  multiplies zero there.

- **Why it passed until now.** `HJBFDMSolver.honors_inhomogeneous_neumann` is `True` and pinned by
  `test_hjb_still_accepts_a_neumann_value`, so the declaration was live. The existing applicator
  test `test_nonzero_neumann_flux_recovered` builds its buffer **with** `domain_bounds` — validating
  the instrument on a path the solver never takes. Found by independent review of #1902.

- **One merged test was re-pointed, and its own precondition is what reported the need.**
  `test_a_tied_wall_row_that_is_not_a_switching_node_still_gets_the_right_branch` (#1896) used the
  Robin `alpha == beta` wall as its witness for "ties in value but is not a switching node". That
  tie was an **artefact of the `dx = 1.0` fallback** and disappears once the real spacing is
  threaded. The witness is now a BC *consistent with the state* — a linear state of slope 2 under
  `neumann(du/dn = 2)`, whose ghost continues the line exactly, so `lap[-1] = 0` while
  `central[-1] = 2`. The test asserted its precondition rather than assuming it, which is the only
  reason this surfaced as a clear failure instead of a silently vacuous pass.

- **Still open in #1904, and larger**: the no-flux ghost is `u[-1] = u[0]`, a *cell-centred* mirror
  on a *node-centred* grid, so the wall Laplacian converges to **half** the true value
  (`0.4959 → 0.499984` over Nx 21…321, against `0.9918 → 0.99997` for the node-centred reflection).
  That is the step that changes the discrete problem for every no-flux solve in the repository, and
  it is what #1902 is blocked on. This change does not touch it.
