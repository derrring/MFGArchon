- **`ghost_cell_fp_no_flux`'s vertex-centred branch used the cell-centred wall geometry, so its
  total-flux residual did not converge** (Issue #2068). It computed
  `rho_g = rho_i·(D + v_n·dx)/(D − v_n·dx)`, which is consistent only with
  `rho_face = (rho_g + rho_i)/2` **and** `drho/dn = (rho_g − rho_i)/(2·dx)` — a wall midway between
  ghost and interior at separation `2·dx`. On a vertex grid the wall **is** the interior node, which
  is why `ghost_cell_dirichlet` returns `g` there unmodified and why `ghost_cell_neumann` uses
  separation `dx` on both centrings (#1972). Corrected to `rho_g = rho_i·(D + v_n·dx)/D`.

  This is the #1972 pathology in a second function: self-consistent with a wrong wall position, so
  it satisfied its own stencil to machine zero while the physical condition went unmet. Measured at
  `D = 0.125`, `rho = 1`, `v_x = +0.5`, max wall:

  | `dx` | old ghost | `J·n` residual | corrected ghost | residual | cell-centred control |
  |-----:|----------:|---------------:|----------------:|---------:|---------------------:|
  | 0.10000 | 2.333333 | −1.166667 | 1.400000 | 1.1e−16 | 3.3e−16 |
  | 0.05000 | 1.500000 | −0.750000 | 1.200000 | 1.1e−16 | −2.2e−16 |
  | 0.02500 | 1.222222 | −0.611111 | 1.100000 | −4.4e−16 | −4.4e−16 |
  | 0.01250 | 1.105263 | −0.552632 | 1.050000 | −4.4e−16 | −4.4e−16 |
  | 0.00625 | 1.051282 | −0.525641 | 1.025000 | 1.8e−16 | 1.8e−15 |

  The old residual settles at `−v_x·rho`, not zero. The cell-centred branch on the same inputs is
  machine zero at every resolution, which is the control that puts the defect in the branch rather
  than in the harness.

  **It also removes a pole.** The old form is singular at `dx = D/v_n` — **half** the cell-centred
  limit `2D/v_n`, because the wrong geometry doubles the effective step — and returns a **negative
  density** past it: `+499` at `dx = 0.249` and `−501` at `dx = 0.251` for the numbers above. The
  corrected form is linear in `dx` and has no pole. It still turns negative under strong *inward*
  drift beyond `dx > D/|v_n|`, which is a resolution requirement rather than a blow-up, and the new
  tests assert that so the difference is not mistaken for unconditional positivity.

  **The pins are external oracles, not the defining equation rearranged.** Asserting
  `v_n·rho_i == D·(rho_g − rho_i)/dx` would restate the fix, which is how the retired convention
  survived elsewhere. Instead: the two centrings discretise the same continuous condition, so their
  leading correction in `dx` must agree — measured 0.99375 for the corrected branch against
  **2.01266** for the retired one at `dx = 3.125e−3`, the factor of two being exactly the "diffusive
  flux is twice what the condition requires" the issue found. Reverting the branch turns six tests
  red; the `drift = 0` case survives, correctly, because every form returns `rho_interior` there.

  **The correction does not make the two centrings equivalent, and the fragment says so rather than
  leaving it to be discovered.** The exact no-flux profile is `rho(x) = rho_i·exp(v_n·x/D)`, so each
  form is a rational approximation to `exp(z)` at `z = v_n·dx/D`. Measured against it:

  | form | as a series | error | measured rate |
  |:-----|:------------|:------|--------------:|
  | retired vertex | `(1+z)/(1−z) = 1 + 2z + …` | `O(dx)` | 1.06 |
  | this branch | `1 + z` | `O(dx²)` | 2.01 |
  | cell-centred | `(1+z/2)/(1−z/2)`, the Padé(1,1) of `exp` | `O(dx³)` | 3.04 |

  The retired form's leading term is `2z` where it must be `z` — it was **inconsistent**, not merely
  coarse. The corrected branch is consistent and remains one order below the cell-centred one,
  because a signature carrying only `interior_value` admits no centred difference at the node. That
  is a property of the interface and is not changed here.

  **BREAKING for `ghost_cell_fp_no_flux(..., grid_type=VERTEX_CENTERED)` and for
  `ZeroFluxCalculator` constructed on a vertex grid** — it returns 1.4 where it returned 2.333333
  on the numbers above. The path is public and exported; it has no consumer in this repository
  (`use_zero_flux=True` is passed nowhere, and `VERTEX_CENTERED` appears in no example, notebook or
  benchmark), so nothing here breaks. Same shape and same file as
  `changelog.d/2064-robin-vertex-branch.fixed.md`.

  On the `2·dx`-separation family: `#1972` fixed `ghost_cells.py`'s Neumann ghost,
  `high_order_ghost_neumann`'s `order<4` arm already raises rather than computing, and this is the
  last live member **in this file**. Two copies remain live in `_compat.py`; #2067 is in review as
  PR #2125 and is not merged, so nothing here should be read as asserting they are fixed.

  Not addressed, and filed rather than folded in: after this change the branch computes the same
  value as `ghost_cell_robin(alpha=v_n, beta=−D, g=0)` to 1e−16 on both centrings, which is the
  duplication class this repository keeps closing. It is not collapsed here because the owner
  *raises* where this function's `1e-12` guard silently returns `interior_value`, and at `D → 0`
  delegating produces a negative density on the cell branch — a behaviour change that deserves its
  own change and its own review.
