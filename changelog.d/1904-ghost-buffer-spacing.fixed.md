- **The ghost buffer no longer falls back to `dx = 1.0`** (Issue #1904, first of three steps).
  `PreallocatedGhostBuffer` derived its spacing only from `domain_bounds`, which almost no caller
  passes, so every consumer read `dx = 1.0` and an inhomogeneous Neumann condition was applied as
  `g/h` instead of `g`. Reading the flux back off the ghost, `-(padded[1] - padded[0])/dx`:

  | Nx | requested `du/dn` | before | after |
  |--:|--:|--:|--:|
  | 21 | 2.0 | 40 | **2.0** |
  | 81 | 2.0 | 160 | **2.0** |
  | 321 | 2.0 | 640 | **2.0** |

  Exactly `g/h` before, exactly `g` after, at both walls and independent of the state. **This fixes
  the flux the ghost encodes, not the accuracy of any derivative built from it** — the centred wall
  gradient the HJB path forms still converges to `g/2 − u'(wall)/2` at `O(h)`, which is #1904's
  node-centring half and is untouched.

  `pad_array_with_ghosts` and `PreallocatedGhostBuffer` now accept an explicit `spacing` (scalar or
  one per axis; a wrong length raises rather than broadcasting), and three call sites that already
  held one now pass it. No-flux and `neumann(0)` are byte-identical, asserted — the flux term
  multiplies zero there.

  **Seven further call sites hold a spacing in scope and still drop it**, and they are live:
  `FDMApplicator.apply` accepts `grid_spacing` and discards it; the advection, divergence and
  gradient operators hold `self.spacings`; two tensor-calculus entry points hold `dx`, `dy`.
  Measured, `PartialDerivOperator` under `neumann_bc(2.0)` on a 21-point unit grid returns ∓20.24 at
  the walls where the exact derivative is `0.0`. Threading them changes returned values on public
  paths and stays with #1904.

  **Separately, and not introduced here**: at the LOW wall the equation the applicator writes is not
  the Robin condition. `robin_bc(g, alpha=0, beta=1)` and `neumann_bc(g)` are the same mathematical
  condition and their low-wall ghosts differ by exactly `2·h·g`, because the Robin branch applies
  `outward_sign` to `(u_g − u_i)/dx`, which is already `du/dn` there. #1262 fixed that sign on the
  NEUMANN branch and not on ROBIN. Threading the spacing rescales that defect rather than curing it.
