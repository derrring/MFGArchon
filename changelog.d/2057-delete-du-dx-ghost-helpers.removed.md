**BREAKING for anyone calling the private helpers.** `BaseStructuredApplicator._compute_ghost_neumann`,
`_compute_ghost_dirichlet` and `_compute_ghost_robin` are removed, with the five tests that were
their only callers (#2057).

They had **zero production callers** — 0, 0, 0 across the package, against a control where
`get_bc_type_for_point` returns 3 — and were called only from tests (4, 5, 1). Their docstring said
they existed to "eliminate duplication ... previously duplicated at `applicator_fdm.py:1099, 1274,
1748`". That duplication *was* eliminated, by `NeumannCalculator` / `ghost_cell_dirichlet` /
`ghost_cell_robin` — different owners. So this was a consolidation that added an owner and never won
its callers, kept green by tests that exercised it in isolation. A test suite cannot notice that a
shared implementation is not shared.

They also ran the opposite convention: measured over slopes, offsets, two spacings and both
centrings, their `g` is `du/dx`, exact at both walls, while `du/dn` fails the min wall and an inward
reading fails the max. The live path (`ghost_cell_neumann`) takes `du/dn`. And the five tests
asserted `expected = u_next_interior - 2*dx*g`, restating the body, so nothing established that the
convention was intended rather than a sign bug.

**This does not remove the `du/dx` convention from the package**, and an earlier draft of this note
claimed it did. `_compat.py`'s `_compute_ghost_pair` still reads `g` as `du/dx` in its Neumann
branch while passing the same `g` to `ghost_cell_robin`, which reads it as `du/dn`, four lines
below — so whichever convention a caller uses, one of those two branches is wrong at the low wall.
That copy is exported, reachable through the package `__getattr__`, and held in place by a
characterization test that restates its arithmetic. #2067 tracks it. What this change removes is the
copy nobody called.

Three stale statements of the retired one-cell Neumann form are corrected, and two more are deleted
with the methods that carried them:

- `BoundaryCalculator`'s protocol summary now gives the Neumann ghost as `u_interior + dx*g` with
  `g = du/dn`, at both walls, and qualifies the Dirichlet row by centring — `ghost_cell_dirichlet`
  returns `2g - u_i` cell-centred and `g` vertex-centred.
- **`applicator_fdm.py`'s module docstring** carried the one-cell form with a two-cell step, in the
  header of the file that houses `FDMApplicator`. Measured on its own stated geometry, `u = 3x`,
  `dx = 0.1`: wrong at **both** walls under **both** readings of `g` — left returned `+0.75` (as
  `du/dn`) or `-0.45` (as `du/dx`) against an exact `-0.15`; right returned `+3.45` against `+3.15`.
  Its defining relation `(u_i - u_g)/(2*dx)` returns `1.5` on a field whose `du/dx` is `3`.
- The census note in `ghost_cells.py` wrongly excluded that site, asserting the occurrence there was
  the two-cell `u_next_interior ± 2*dx*g` — a string that appears zero times in that file. The note
  no longer carries a count at all: two revisions of it were right for one spelling and wrong for
  the population, and `u_ghost = u_interior + 2*dx*g` versus `u_g = u_i ± 2*dx*g` is exactly how the
  excluded site hid. A tally over a hand-chosen literal cannot audit the predicate that chose it.
