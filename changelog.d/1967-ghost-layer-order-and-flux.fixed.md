**The high wall paired its ghost layers backwards, and an inhomogeneous flux offset did not grow
with the layer.** Two defects in the same loop, in both copies of it.

The high ghosts occupy padded indices `-g .. -1` and `-g` is the one adjacent to the wall, so both
the ghost walk and the interior walk must start there. The high-wall loop started its ghost walk
at `-1`, the far end, while its interior walk started near — pairing the layers in reverse for
`ghost_depth >= 2`. Measured on `cos(2πx)`, which is even about both walls so `NEUMANN(0)` is
exact at both: the low wall was machine-zero at every depth while the high wall was **5.4e-01 at
`g=2` and 1.3e+00 at `g=3`**, with `reversed(got) == want` — every value correct, every slot wrong.

Separately, ghost layer `k` sits `(2k-1)·dx` from its mirror, so an inhomogeneous Neumann offset
must scale with the layer. `dx · v` was applied to every layer — the `k = 1` value used
throughout, drifting by `(2k-2)·dx·v`.

**Both are invisible at `ghost_depth = 1`**, where a single layer has no order to reverse and
`(2·1−1)·dx = dx`. Every caller in the library passes 1 or omits it, which is why a suite of 6147
was green with both present. The one production consumer at depth 3 is `hjb_weno.py:330`, whose
`_SUPPORTED_BC_TYPES` is `{NEUMANN, NO_FLUX, PERIODIC}` — exactly the family the first defect hits.

Eighteen of twenty-four probed cells are unchanged, every `ghost_depth = 1` cell among them, and
that is pinned: since every caller uses depth 1, a change there would be a regression in
everything that currently works.

Found by four independent verification oracles — convergence order, invariance,
cross-implementation and grid convention — run in parallel on #1966, which had certified this
family as correct after measuring one wall.
