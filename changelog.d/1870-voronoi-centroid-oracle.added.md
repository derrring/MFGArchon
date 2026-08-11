- **A closed-form oracle for the clipped-Voronoi cell centroid** (PR #1870). `CellGeometry.centroid`
  is populated for every cell and read by nothing in this package — its consumer is `mfg-research`,
  whose own pin sits behind a `skipif` on the field's existence and so self-skips exactly when it
  would matter. Measured before this file existed: halving every centroid
  (`/ (3.0 * twice_area)` → `/ (6.0 * twice_area)`) left the full local gate at
  `5960 passed ... GATE GREEN`, byte-identical. The implementation is correct; nothing was checking.
  The oracle is external in the sense the close-out policy means — the area centroid of a polygon has
  a closed form, computed independently of the implementation rather than captured from it, so "there
  is no oracle for this yet" would have been false. Five shapes including a non-convex L and one off
  the origin, plus translation equivariance, a bounding-box containment check over every cell the
  public path produces, and an area-weighted centroid over a symmetric tiling.
  Mutation-verified, four ways: halving every centroid and dropping the offset term each turn 8
  assertions red; swapping `x` and `y` turns 4 red and is caught only by the asymmetric shapes, which
  is the point of including them; removing the degeneracy guard turns exactly the test written for it
  red.
- **The degeneracy guard is tested directly, because it is unreachable from the caller.**
  `clipped_voronoi_cells` refuses a cell at `area <= 1e-14`, and `_polygon_centroid`'s own threshold
  is `abs(2A) <= 2e-14` — the same bound on a CCW polygon, checked afterwards. So within the library
  the `ValueError` is a branch guarding a state its only caller has already refused. Testing it
  directly is the alternative to deleting it as defensive code for an impossible state; which of
  those is right depends on whether the helper is meant to be callable on its own, and the test says
  which behaviour is being relied on either way.
