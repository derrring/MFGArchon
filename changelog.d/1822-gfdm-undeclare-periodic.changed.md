`HJBGFDMSolver` and `FPGFDMSolver` no longer declare `BCType.PERIODIC`. Neither ever honoured it,
at any grid size tried:

| Nx | `HJBGFDMSolver` | `FPGFDMSolver` |
|---:|:--|:--|
| 11 | `NotImplementedError` at boundary point 0 | density → −3.83e-01 at t=6 |
| 21 | same | density → −1.03e+00 at t=4 |
| 41 | same | density → −1.88e-01 at t=3 |

The HJB raise's own message said *"Use TensorProductGrid + FDM for periodic geometries"* while the
class advertised the type, and the declaration carried a comment claiming a working "ghost-skip"
path for uniform periodic — there is none; the raise fires for uniform periodic every time. The FP
side fails **earlier** as the grid refines, so it is not a resolution problem either.

A caller passing `periodic_bc` to these solvers now gets a clear refusal from the BC-capability gate
at construction, instead of a `NotImplementedError` from deep in a row builder or a negative density
several timesteps into a solve. Nothing that ever worked is lost.

**The declared-surface matrix was re-keyed in the same change, and that is the load-bearing part.**
`_surface_params` iterated the PERIODIC-declaring set, which looked like no filter at all because
every searched class declared PERIODIC. Undeclaring it would have silently deleted five rows —
three recording live defects (`FPGFDMSolver-NEUMANN`, `FPGFDMSolver-NO_FLUX`,
`HJBGFDMSolver-DIRICHLET`) — while the file stayed green: measured, 0 GFDM rows collected and
34 passed / 10 xfailed, against 5 rows and 37 / 13 now. The surface matrix now iterates every
solver declaring *any* BC type; the seam and mass invariants stay keyed on PERIODIC, which is
correct for them since they are statements about periodicity.

`test_the_surface_matrix_measures_every_declared_pair_it_has_a_fixture_for` asserts that coverage
directly, and reddens under exactly the narrowing described above — a pair that is absent from a
matrix reads identically to a pair that passed.
