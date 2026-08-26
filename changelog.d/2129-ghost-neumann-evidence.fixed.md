- **`ghost_cell_neumann`'s docstring cited a linear field as the evidence for its wall geometry, and
  a linear field cannot show it** (Issue #2129). The line read *"verified exact on 12 combinations
  (2 centrings × 2 walls × slopes 3, −1.7, 0) against `u = a*x`"*. That is true — and it is equally
  true of the `2·dx` mirror `u_next ∓ 2·dx·(du/dx)` that #1972 replaced: on a linear field the two
  are byte-identical at both centrings and both walls, all six pairs. Reproducing a linear field is
  the weakest thing a first-order ghost rule can do, so passing that check separated nothing.

  **What discriminates is not the ghost value but the derivative the consuming stencil takes from
  it**, and that consumer is a centred difference —
  `operators/stencils/finite_difference.py:95` computes `(u[i+1] − u[i−1]) / (2h)`. Measured on
  `u = sin x + 0.3x²` with `du/dn` prescribed at the wall, error in `u'` at the wall:

  | centring | shipped `u_int + dx·g` | retired `2·dx` mirror |
  |:---------|:-----------------------|:----------------------|
  | **cell** | `0.0` at every `dx` — **exact** | first order |
  | **vertex** | rate 0.96, 0.98, 0.99 — **`O(dx)`** | exact |

  Cell-centred puts the ghost at `−dx/2` and the interior at `+dx/2`, so the centred difference
  across them is centred **on the wall** and the shipped form makes it exact. Vertex-centred puts
  the wall on the interior node, so the same difference spans `2·dx` around it and the one-sided
  ghost leaves `O(dx)`. **The two forms are order mirror images**, which is why no measurement
  symmetric in the two could ever have chosen between them.

  **One formula is still kept, and the reason is now written down** — it is a fact about this
  package, not about the mathematics. `VERTEX_CENTERED` appears **zero** times in `alg/` and
  `solvers/`, so no scheme here reaches the vertex arm while the cell arm is on every live path.
  Two formulas would mean two owners and a `grid_type` argument on a function that needs neither,
  to serve a configuration nothing requests. That trade is recorded rather than assumed, and the
  line names #2129 as where to start if a vertex-grid scheme ever lands.

  `test_ghost_neumann_evidence_2129.py` pins both halves — the negative one so the citation is not
  restored as if it settled something, and the positive one so the property it was meant to
  establish has a measurement behind it. It also asserts that the retired mirror **is** exact at a
  vertex wall, which is what makes this a recorded trade rather than a defect left standing.

  `high_order_ghost_neumann`'s docstring quoted `ghost_cell_neumann` at "rate 3.00 → O(h³)" without
  naming a centring; that is the cell-centred rate, and it now says so.
