Three docstrings in `geometry/boundary/protocols.py` stated `u_ghost = u_interior + 2*dx*g`, the
one-cell Neumann form #1972 removed from `ghost_cell_neumann`'s body (#2057). Two distinct errors,
both corrected:

`BoundaryCalculator`'s protocol summary describes the Calculator layer, which resolves to
`NeumannCalculator` → `ghost_cell_neumann` → `interior + dx*g` with `g = du/dn`. It now says that.

`BaseStructuredApplicator._compute_ghost_neumann`'s derivation named the wrong variable and
contradicted its own next line: it wrote `u_ghost = u_interior + 2*dx*g` while the zero-flux line
two lines below said `u_ghost = u_next_interior`, and the body has always used `u_next_interior` —
reflecting across the boundary from two cells in, which is what makes the step `2*dx` there and
`dx` in `ghost_cell_neumann`.

It also labelled its flux `du/dn`. Measured on `u = 3x`, `dx = 0.1`: under `du/dx` both walls are
exact; under `du/dn` the left wall returns `+1.05` where `-0.15` is exact. **The parameter is
`du/dx`, the opposite of the convention its live sibling uses.** The docstring now says so and
points at #1936, which owns the two-conventions problem rather than resolving it here.
