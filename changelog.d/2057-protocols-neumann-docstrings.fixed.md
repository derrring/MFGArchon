**BREAKING for anyone calling the private helpers.** `BaseStructuredApplicator._compute_ghost_neumann`,
`_compute_ghost_dirichlet` and `_compute_ghost_robin` are removed, with the five tests that were
their only callers (#2057).

They had **zero production callers** — 0, 0, 0 across the package, against a control where
`get_bc_type_for_point` returns 3 — and were called only from tests (4, 5, 1). Their docstring said
they existed to "eliminate duplication ... previously duplicated at `applicator_fdm.py:1099, 1274,
1748`". That duplication *was* eliminated, by `NeumannCalculator`, a different owner. So this was a
consolidation that added an owner and never won its callers, kept green by tests that exercised it
in isolation — a test suite cannot notice that a shared implementation is not shared.

Worse, it ran on the opposite convention: measured over slopes, offsets, two spacings and both
centrings, its `g` is `du/dx`, exact at both walls, while `du/dn` fails the min wall and an inward
reading fails the max. The live path (`ghost_cell_neumann`) takes `du/dn`. A caller moving between
them would have been right at one wall and silently sign-flipped at the other. Deleting the
`du/dx` side removes the second convention rather than documenting it.

Three stale statements of the retired one-cell Neumann form `u_ghost = u_interior + 2*dx*g` are also
corrected:

- `BoundaryCalculator`'s protocol summary now gives the Neumann ghost as `u_interior + dx*g` with
  `g = du/dn`, at both walls, and qualifies the Dirichlet row by centring — `ghost_cell_dirichlet`
  returns `2g - u_i` cell-centred and `g` vertex-centred.
- **`applicator_fdm.py`'s module docstring** carried the one-cell form with a two-cell step, in the
  header of the file that houses `FDMApplicator`. Measured on its own stated geometry, `u = 3x`,
  `dx = 0.1`: wrong at **both** walls under **both** readings of `g` — left returned `+0.75` (as
  `du/dn`) or `-0.45` (as `du/dx`) against an exact `-0.15`; right returned `+3.45` against `+3.15`.
  Its defining relation `(u_i - u_g)/(2*dx)` returns `1.5` on a field whose `du/dx` is `3`.
- The census note left in `ghost_cells.py` by #2055 **wrongly excluded that site**, claiming the
  occurrence in `applicator_fdm.py` was the two-cell `u_next_interior ± 2*dx*g` form — a string that
  appears zero times in that file. A miscounted exclusion shielded the very site the count existed
  to find. The same note said the user guide still carried the form; #2055 had corrected the guide
  in the same commit. Both corrected.
