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
- **The degeneracy guard is tested directly, and the "unreachable" claim is narrowed to what was
  searched.** `clipped_voronoi_cells` refuses at `area <= 1e-14` and the helper's threshold is
  `abs(2A) <= 2e-14`, the same bound — except that the caller reverses a negatively-oriented polygon
  first, and reversal changes the shoelace sum's summation order. Searched for a polygon that passes
  one and trips the other: 320k random over extents `1e-8..1e-6`, offsets `0..1e7` and 3..9 vertices,
  plus 45 deliberate slivers of length up to `1e4` and thickness to `1e-17` — none found, largest
  reversal asymmetry `4.0e-28` against a `2e-14` threshold. A review reported three out of 5106;
  that has not been reproduced and the disagreement is recorded rather than resolved. The test makes
  the branch live either way.
- **Two wrong centroids escaped the first version of this file and are now caught.** Replacing the
  call site with `poly.mean(axis=0)` — the vertex mean, which differs from the area centroid by
  `3.5e-02` on an irregular cloud — passed all nine assertions, because the closed-form tests call
  the helper directly and the two that use the public path are satisfied by any convex combination
  of a cell's vertices, on a symmetric tiling, always. And taking `abs` of the signed area alone
  negates the result for clockwise input, which every shape in the file avoided being. Both now have
  a test written for them; the mutation battery is six and all six die.
