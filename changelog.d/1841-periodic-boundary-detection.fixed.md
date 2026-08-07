- **`HJBGFDMSolver` solves a periodic problem from an ordinary collocation cloud** (Issue #1841,
  refs #1822, #1840, #711). The capability was always there — seam 2.2e-15 / 3.3e-11 / 6.7e-16 /
  6.7e-16 at Nx=11/21/41/81, three orders under the #1822 matrix's `SEAM_TOL = 1e-9` — but reachable
  only by hand-passing `boundary_indices=np.array([], dtype=int)`, i.e. lying to the solver about its
  own geometry. Three things stood between the default cloud and that path, and all three are fixed:
  boundary detection classified a point on a *periodic* face as a boundary point (a periodic axis has
  no boundary — its two faces are the same physical place), so the point reached a row builder with
  no periodic row and raised; no periodic wrap was constructed at all unless a separate periodic
  collocation geometry was supplied, so the seam would have stayed open even once detection was
  fixed; and the wrap's own guard tested `isinstance(geometry, SupportsPeriodic)` and then called
  `geometry.wrap_displacement(...)`, **a method that protocol does not declare** — so a periodic
  `TensorProductGrid` passed the guard and raised `AttributeError`.
  That last one is fixed by consolidation rather than by adding the missing method. `SupportsPeriodic`
  already promises `get_periods()`, both geometries implement it, and the minimum-image wrap needs
  nothing else — so the arithmetic moves to `geometry/boundary/periodic.py`, which already owns
  `wrap_positions` and `create_periodic_ghost_points`, and the single caller derives it from
  protocol-declared data. `Hyperrectangle.wrap_displacement` is deleted: implementations of the DISPLACEMENT
  form go 2 → 1 and calls to the method go 1 → 0. The same minimum-image rule still has two
  further implementations as `compute_periodic_distance`, one per geometry, and independent
  review measured both to be wrong for `|delta| > 1.5L` — 0.9 where the answer is 0.1. Out of
  scope here and filed as #1853, so the honest count of implementations in the tree is three. Behaviour is pinned byte-for-byte against output
  captured **before** the change (5 cases, including the no-periodic-dims identity path and the 1-D
  input branch), because after consolidation an A-vs-B comparison would be tautological.
  The two #1822 rows come off `xfail`: 43 passed / 13 xfailed → 45 / 11.
