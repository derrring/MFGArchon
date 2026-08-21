**BREAKING.** `high_order_ghost_neumann` and `high_order_ghost_dirichlet` are retired and now raise
`NotImplementedError` (#1936). Both shipped in v0.21.0 (public since #849, 2026-03-28), both were
called by nothing in the package or the tests, and neither delivers the order in its name.

Neither is uniformly wrong, which is why reading them did not settle it. `high_order_ghost_neumann`
uses `flux_value * outward_normal_sign` as the **inward** derivative: at the min wall that coincides
with the truth and `order=4`/`order=5` are exact, while at the max wall — the parameter's default —
they are off by exactly `12hg/11` and `24hg/25`. Both high-order branches separately impose the
derivative constraint at the ghost centre rather than at the face, which a linear field cannot
expose and `u = x^2` with `g = 0` does (0.5455, 0.4800). Together these cost the advertised order:
on `u = exp(x)` the ghost value converges at rate 1.01 where `ghost_cell_neumann` gives 3.00, so the
"high-order" routine is first-order and worse than the second-order rule it was written to improve
on. Its `order<4` fallback is a different failure — `u[0] + 2*dx*g` is the pre-#1972 formula struck
in this same file on 2026-08-18, surviving in a copy nothing checked.

`high_order_ghost_dirichlet` cannot reproduce a constant. The prescribed value is itself a value of
`u` at the face, so the coefficients must sum to 1; on `u = 1` its **default** cell-centred path
returns `[1.6, 4.0]` at `order=4` and `[1.5, 3.3333]` at `order=5` where `[1.0, 1.0]` is required.
Its vertex-centred branches and its `order<4` fallback are correct.

Both names still resolve and each error names its replacement — `ghost_cell_neumann` and
`ghost_cell_dirichlet`. No high-order ghost capability is lost that was ever there. A correct
face-constrained 4-point Neumann formula is recorded on #1936 rather than implemented here: that
issue's acceptance criterion is that the ghost value gain a single owner and the implementation
count drop, and a sixth implementation serving zero consumers moves it the wrong way.
