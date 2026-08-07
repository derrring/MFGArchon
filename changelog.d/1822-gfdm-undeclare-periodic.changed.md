- **`FPGFDMSolver` no longer declares `BCType.PERIODIC`, and the declared-surface matrix stopped
  being scoped to periodicity** (Issue #1822, refs #1841). The solver never had a periodic path:
  it builds its `TaylorOperator` with no `geometry=`, so no periodic wrap is ever constructed, and
  the `"periodic"` its BC resolver returns is read by nothing downstream. Declaring it bought a
  caller a `ValueError` several timesteps into a solve — the density goes negative, and sooner as
  the grid refines, so not a resolution problem — where the #1456 capability gate now refuses at
  construction.

  **`HJBGFDMSolver` keeps its declaration.** An earlier version of this change removed it too, on
  the grounds that there was "no working path at any grid size", and that was measured on the
  default collocation cloud only. Give it a cloud whose points are not *detected* as boundary plus
  a periodic `collocation_geometry`, and the Issue #711 wrap does the work: seam 2.2e-15, 3.3e-11,
  6.7e-16, 6.7e-16 at Nx=11/21/41/81, against this campaign's own `SEAM_TOL = 1e-9`. The capability
  is real; only its reachability is broken, because `_detect_boundary_indices` ignores
  `periodic_dims` and routes a point on a face to a row builder that has no periodic row. That is
  #1841, and it is why the declaration and its xfail both stay.

  **The surface matrix was re-keyed in the same change, and that part is load-bearing regardless of
  which solver is undeclared.** `_surface_params` iterated the PERIODIC-declaring set, which looked
  like no filter at all because every searched class declared PERIODIC — so the first solver to
  narrow its declaration falls out of the matrix entirely, taking its unrelated rows with it.
  Measured here: with the old keying, 4 GFDM rows are collected instead of 6, and the two that
  disappear are `FPGFDMSolver-NEUMANN` and `FPGFDMSolver-NO_FLUX`, both recording live defects. The
  matrix now iterates every solver declaring *any* BC type; the seam and mass invariants stay keyed
  on PERIODIC, which is correct for them since they are statements about periodicity.
  `test_the_surface_matrix_measures_every_declared_pair_it_has_a_fixture_for` asserts that coverage
  directly and reddens under exactly that narrowing — a pair absent from a matrix reads identically
  to a pair that passed.
