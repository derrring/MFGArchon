- **`levy_integro_diff.py`'s "Tests MUST include non-uniform grid adjoint consistency" is enforced
  again** (Issue #2268). The test satisfying that constraint was deleted by `18d8cc80` (#2227),
  leaving live source asserting a property nothing checked. Restored as the binding half only — the
  discrete duality `<J[v], m>_W == <v, J*[m]>_W` on a non-uniform grid — rather than the whole
  deleted file, whose remaining cases were happy-path assertions the #2227 bar removed on purpose.
  Admitted as an external oracle: it pins the mathematics of the `L^2(W)` adjoint, computed from the
  operator's published `grid` and `integration_weights`, so it does not rot when the API moves.

  Two things the restoration measured and the constraint's own wording does not imply. A `linspace`
  grid does **not** give `W ∝ I` — trapezoidal weights halve both endpoints, so the W-free transpose
  `J^T` fails there too, by order 1; and restricting the pairing to interior indices does not rescue
  it, because `(W^{-1} J^T W m)_i` sums over every `j`. What the non-uniform grid uniquely buys is
  **interior** weight variation (ratio 2.19 against 1.00 for `linspace`), which is what an
  implementation assuming constant spacing would get wrong while still handling endpoints.
