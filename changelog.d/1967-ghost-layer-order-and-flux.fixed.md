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
`(2·1−1)·dx = dx`. `hjb_weno.py:330` is the library's only production `ghost_depth >= 2`, so every
other caller sits in that blind spot, which is why a suite of 6147 was green with both defects
present.

**Reachability, measured rather than inferred.** WENO's `_SUPPORTED_BC_TYPES` is
`{NEUMANN, NO_FLUX, PERIODIC}` — the family these defects hit — but on every *uniform* BC it
admits, it dispatches to `_apply_poly_extrapolation`, which neither defect touches and this change
leaves byte-identical. The corrected lines are reached only through a *per-face* BC, and no
in-repo code hands WENO one: the two per-face HJB constructors both emit `BCType.ROBIN`, which
WENO refuses at construction. So this is a correctness fix on a path that is reachable but not
currently reached — not, as an earlier draft of this fragment said, the live consumer's own path.

Eighteen of twenty-four probed cells are unchanged, every `ghost_depth = 1` cell among them, and
that is pinned: since every caller uses depth 1, a change there would be a regression in
everything that currently works.

Found by four independent verification oracles — convergence order, invariance,
cross-implementation and grid convention — run in parallel on #1966, which had certified this
family as correct after measuring one wall.

A **third** copy of the same reversal survives in `GhostBuffer._update_bounded`, at both walls and
with identical magnitudes; it is unreachable today (`create_ghost_buffer_from_bc` has no library
caller) and is filed separately. This change corrects two of three.
