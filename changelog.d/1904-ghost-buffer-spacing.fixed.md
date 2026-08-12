- **The ghost buffer no longer falls back to `dx = 1.0`** (Issue #1904, first of three steps).
  `PreallocatedGhostBuffer` derived its spacing only from `domain_bounds`, which almost no caller
  passes — `pad_array_with_ghosts(..., geometry=None)` left `_grid_spacing = None` and every
  consumer read `dx = 1.0`. An inhomogeneous Neumann condition was therefore applied as `g/h`
  instead of `g`. Reading the flux back off the ghost, `-(padded[1] - padded[0])/dx`:

  | Nx | requested `du/dn` | before | after |
  |--:|--:|--:|--:|
  | 21 | 2.0 | 40 | **2.0** |
  | 41 | 2.0 | 80 | **2.0** |
  | 81 | 2.0 | 160 | **2.0** |
  | 161 | 2.0 | 320 | **2.0** |

  Exactly `g/h` before, exactly `g` after, at both walls and independent of the state.
  **That expression is the algebraic inverse of the ghost write, so it certifies what the ghost
  encodes and not the accuracy of any derivative built from it** — the centred wall gradient the
  HJB path forms converges to `g/2 - u'(wall)/2` at O(h), which is the node-centring half of #1904
  and is untouched here.

- **Robin read the same fallback**, in the denominator `alpha + beta/dx`. Its ghost is the exact
  algebraic solution of the equation the applicator writes, so the wall residual is machine-zero
  (`<= 1e-13`) for every `alpha` and `beta`, at both walls, once the spacing is real — measured
  over `beta` in {0.25, 0.5, 1, 2, 3.7} at Nx 21/81/321.
  ~~exactly 0 for `beta = 1` and O(h) otherwise (`2.5e-03 → 6.3e-04 → 1.6e-04`)~~ was in this
  fragment and does not reproduce under any convention tried: there is no `beta`-dependence to
  find. **Separately, and not introduced here**: at the LOW wall the equation the applicator writes
  is not the Robin condition. `robin_bc(g, alpha=0, beta=1)` and `neumann_bc(g)` are the same
  mathematical condition and their low-wall ghosts differ by exactly `2*h*g` (the high wall agrees
  to 1e-16), because the Robin branch applies `outward_sign` to `(u_g - u_i)/dx`, which is already
  `du/dn` there. `#1262` fixed exactly this sign on the NEUMANN branch and not on ROBIN. Filed
  separately; threading the spacing rescales that defect rather than causing or curing it.

- **The spacing was already in hand at every site that needed it — and three of them now pass it.**
  `laplacian_with_bc` takes `spacings` as a parameter and dropped it at the padding call;
  `_compute_gradient_array_1d` has `Dx`. `pad_array_with_ghosts` and `PreallocatedGhostBuffer` now
  accept an explicit `spacing` (scalar or one per axis, wrong length raises rather than
  broadcasting), and `finite_difference.py:445`, `:492` and `base_hjb.py:87` pass it. No-flux and
  `neumann(0)` are byte-identical, asserted — the flux term multiplies zero there.

  **Seven further call sites hold a spacing in scope and still drop it**, and they are live rather
  than theoretical: `applicator_fdm.py:203` (`FDMApplicator.apply`, which accepts `grid_spacing`
  and discards it), `advection.py:261` and `:265`, `divergence.py:177`, `gradient.py:181` (all four
  hold `self.spacings`), and `tensor_calculus.py:1049` and `:1068` (hold `dx`, `dy`). Measured:
  `PartialDerivOperator` under `neumann_bc(2.0)` on a 21-point unit grid returns $\mp 20.24$ at the
  walls on a field whose exact derivative there is `0.0`. Threading them changes returned values on
  public paths and belongs in one measured change; it stays with #1904.

- **Why it passed until now.** `HJBFDMSolver.honors_inhomogeneous_neumann` is `True` and pinned by
  `test_hjb_still_accepts_a_neumann_value`, so the declaration was live. The existing applicator
  test `test_nonzero_neumann_flux_recovered` builds its buffer **with** `domain_bounds` — validating
  the instrument on a path the solver never takes. Found by independent review of #1902. The
  declaration is still not fully truthful: on the mixed / non-uniform BC path the NEUMANN branch
  applies a pure mirror and never reads its segment value at all, which no amount of spacing
  threading reaches. Filed separately.

- **One merged test was re-pointed twice, and the second time by review.**
  `test_a_tied_wall_row_that_is_not_a_switching_node_still_gets_the_right_branch` (#1896) used the
  Robin `alpha == beta` wall as its witness for "ties in value but is not a switching node". That
  tie was an **artefact of the `dx = 1.0` fallback** and disappears once the real spacing is
  threaded — the test asserted its precondition rather than assuming it, which is the only reason
  this surfaced as a failure instead of a silently vacuous pass. ~~A slope +2 state read at the
  high wall~~ replaced it and was itself non-discriminating: `central[-1] = +2 > 0` is the half on
  which the tie-break and the tie-agnostic rule agree, so deleting the tie-break left the test
  green while reddening four unrelated interior-row cases. The witness is now slope **-2** read at
  the **low** wall — `lap[0] = 0`, `central[0] = -2` — and the sign of `central` is asserted, not
  assumed, because that sign is what makes the test discriminate. Verified: deleting the tie-break
  now reddens it, at `4.000e+01`.

- **Still open in #1904, and larger**: the no-flux ghost is `u[-1] = u[0]`, a *cell-centred* mirror
  on a *node-centred* grid, so the wall Laplacian converges to **half** the true value
  (`0.4959 → 0.499984` over Nx 21…321, against `0.9918 → 0.99997` for the node-centred reflection).
  That is the step that changes the discrete problem for every no-flux solve in the repository, and
  it is what #1902 is blocked on. This change does not touch it. Measured against that step: 8 of
  the 14 cases added here encode the cell-centred convention and will need re-pointing when it
  lands; the #1896 witness above survives it unchanged.
