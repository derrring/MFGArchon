**BREAKING.** `ghost_cell_robin` no longer takes `outward_normal_sign`, and
`ghost_cell_fp_no_flux` no longer defaults it (#2063, #1936 item 3). The two functions get
opposite treatment because the parameter was doing opposite things in them.

**`ghost_cell_robin`: the argument was wrong, not merely dangerously defaulted.** It was unused in
the cell-centred branch and applied *backwards* in the vertex-centred one. `RobinCalculator` was
the only caller in the package that passed it — AST census over every call site, with the
parameter's index read from the signature — and passing the physically correct `-1.0` at the min
wall is exactly what broke it. Measured through the public calculator on `u = 3x`, `dx = 0.1`,
`alpha = 0, beta = 1` (which makes Robin the same condition as Neumann, so a linear field must come
back exactly):

| grid_type | side | was | exact |
|---|---|---|---|
| CELL_CENTERED | max / min | +3.3000 / −0.3000 | ✓ |
| VERTEX_CENTERED | max | +3.3000 | ✓ |
| **VERTEX_CENTERED** | **min** | **+0.3000** | **−0.3000** |

A sign inversion, not a truncation error. The seven callers that omitted the argument took the
`+1.0` default and were right by accident; they are now right by construction. Ignoring the
argument entirely is exact on 8 of 8 — two centrings × two walls × two slopes — which is why it is
deleted rather than made required. Same removal and same reason as #1972's from
`ghost_cell_neumann`: `du/dn` already carries the wall's direction.

Nothing caught this because the cell-centred branch never used the parameter and `grid_type`
defaults to cell-centred, so every default-taking path was doubly insulated and the one path that
could reach the bug had no test. It has one now.

**`ghost_cell_fp_no_flux`: the parameter stays, the default goes, and the docstring was inverted.**
Here it does real work — it reconciles `ghost_cell_advection_diffusion_no_flux`, which holds `v·n`
and passes `1.0` deliberately, with `ZeroFluxCalculator`, which holds the axis velocity `v_x` and
passes the wall's sign. Both callers pass it and **none relied on the default**, so requiring it is
free and removes a `+1.0` that silently means "max wall".

Its `Args:` line said `drift_velocity` is "the normal component v*n (positive = outward)" while the
body multiplies by `outward_normal_sign` to *obtain* `v·n`, and its own `Example` passes `-0.5` for
a leftward drift at a left wall — where `v·n` would be `+0.5`. The body and the Example agree; the
`Args:` line was the outlier. Decided by the function's own contract, `J·n = 0`: fed `v_x` the
residual is machine zero at both walls; fed `v·n` the min wall leaves **1.25**. Corrected.

The `Args:` entry for `outward_normal_sign` is corrected too. It read "+1 for max boundary, −1 for
min boundary", as though the parameter identified the wall; it does not.
`ghost_cell_advection_diffusion_no_flux` passes `+1.0` at **both** walls because it already holds
`v·n`. The parameter is the conversion factor, and reading it as a wall identifier is what made the
`drift_velocity` line say `v·n` in the first place.

`RobinCalculator` and `ZeroFluxCalculator` now pass `grid_type` by **keyword**. Passed positionally
it lands on whatever slot follows `dx`, so reintroducing an `outward_normal_sign` parameter would
silently rebind `GridType` onto it and fall back to cell-centred — mutation-tested during review:
that restoration leaves 51/51 green. The keyword makes the binding structural rather than lucky.

Not fixed here, and filed while validating the lines this change edits:

- **#2064** — the vertex-centred `alpha != 0` arm of `ghost_cell_robin` is wrong at both walls
  independently of any sign, returning −10.5 where 3.3 is exact.
- **#2068** — `ghost_cell_fp_no_flux`'s VERTEX_CENTERED branch leaves a total flux that does not
  converge under refinement, settling at `−0.5 = −v_x·rho` where the cell-centred branch is machine
  zero at every resolution. It is self-consistent with a wall at separation `2*dx`, which is the
  #1972 pathology in the same file. **The `J·n = 0` test this change adds exercises only the
  cell-centred branch** — the one that already satisfied the contract; replacing the vertex formula
  with arbitrary values leaves 1309/1309 green.
- **#1907**, which recorded the cell-centred symptom of the Robin sign, no longer reproduces —
  re-measured with its own harness, 0.000000 where it recorded 0.2 / 0.05 / 0.0125. It already did
  not reproduce before this change; nothing here fixes it.
