`FPGFDMSolver` no longer declares `BCType.PERIODIC`. It never had a periodic path: it builds its
`TaylorOperator` with no `geometry=`, so no periodic wrap is ever constructed, and the `"periodic"`
its BC resolver returns is not read by anything downstream. Every configuration tried fails —
`taylor_order` 1/2/3, `weight_function` wendland/gaussian, `delta` 0.05/0.3, `upwind_scheme`
none/linear/exponential — with the density going negative mid-solve: −3.83e-01 at t=6 (Nx=11),
−1.03e+00 at t=4 (Nx=21), −1.88e-01 at t=3 (Nx=41), i.e. *sooner* on finer grids, so not a
resolution problem. A caller now gets a refusal from the #1456 capability gate at construction
instead of a `ValueError` several timesteps in.

**`HJBGFDMSolver` keeps its declaration.** An earlier version of this change removed it too, on the
grounds that there was "no working path at any grid size". That was measured on the default
collocation cloud only, and it is false: with a cloud whose points are not *detected* as boundary —
cell centres, or an endpoint-inclusive cloud with an empty `boundary_indices` — plus a periodic
`collocation_geometry`, the Issue #711 wrap gives a seam of **exactly 0.000e+00 at Nx=11/21/41/81**,
against this campaign's own `SEAM_TOL = 1e-9`. The capability is real; only its reachability is
broken. `_detect_boundary_indices` ignores `periodic_dims`, so points on a face are classified as
boundary even on a periodic axis and routed to a row builder that has no periodic row. That is the
defect, and it is #1841.

**The declared-surface matrix was re-keyed in the same change, and that part is load-bearing
regardless of which solver is undeclared.** `_surface_params` iterated the PERIODIC-declaring set,
which looked like no filter at all because every searched class declared PERIODIC. Undeclaring one
solver would silently delete its other rows. Measured with the old keying: **0 GFDM rows collected
and the file still green at 34 passed / 10 xfailed**, against 5 rows and 37 / 13 now — three of
those rows recording live defects (`FPGFDMSolver-NEUMANN`, `FPGFDMSolver-NO_FLUX`,
`HJBGFDMSolver-DIRICHLET`). That is the shape this campaign keeps finding: a check that reads as
comprehensive because the population makes its condition vacuous. The surface matrix now iterates
every solver declaring *any* BC type; the seam and mass invariants stay keyed on PERIODIC, which is
correct for them since they are statements about periodicity.
`test_the_surface_matrix_measures_every_declared_pair_it_has_a_fixture_for` asserts that coverage
and reddens under exactly the narrowing above — a pair absent from a matrix reads identically to a
pair that passed.
