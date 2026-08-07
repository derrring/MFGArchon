`FPFDMSolver` and `FPFVMSolver` now solve on the torus the grid actually describes. Both wrapped
cell `N-1` to cell `0`, which on the endpoint-inclusive grid `TensorProductGrid` builds treats the
repeated endpoint as its own cell and puts the solve on a torus of length `1 + dx`. Against the
analytic heat kernel at `Nx=21` that was 8.7e-02 of relative error; it is now 9.3e-03, converging
to 2.4e-03 at `Nx=81`. Nothing raised, the density stayed positive, and under a datum symmetric
about the midpoint the periodic seam stayed at 2e-15 while the mode's amplitude came out 8.7% off
the analytic value (a decay rate 9.4% low) -- so the `#1822` seam invariant could not see it, and
the new pin is against the heat kernel and a rigid translation instead
(`tests/unit/test_alg/test_periodic_torus_oracle_1822.py`).

The same wrap reached three further paths, each of which would otherwise have disagreed with the
fixed ones: a periodic BC supplied through the constructor or through `components` rather than
through the geometry (both solvers resolve the BC themselves, shadowing the base-class property
that binds the grid's layout), and `AdvectionOperator`'s wrap face, which the callable-`drift_field`
route uses -- that route returned a seam of 3.9e-02 from exactly periodic data and lost 0.54% of
the torus mass.

The wrap arithmetic had 12 live copies across the four FP-FDM advection schemes, the ghost
applicator and the Laplacian assembly; all now route through
`boundary.conditions.periodic_axis_span` and `boundary.types.repeated_endpoint_count`, and the
copies are deleted. Four more remain in `FPFDMSolver._solve_fp_1d`, which has no callers and is
already scheduled for removal by v0.25.0; they are left untouched rather than changed unverified.

Three mass-conservation tests measured mass with a rectangle rule that counts the shared node twice
(it read 1.035 for a density of mass 1 even at `t=0`, so it was never measuring mass on this grid)
and now use the quadrature `invariants.mass_drift` already owned; the same solves conserve to
1.2e-14 under it, and the discrepancy the rectangle rule reported equals `m[0]*dx` to ten digits.

Unstated conventions are untouched, so no caller that never met a grid has its numbers moved:
`bc=None` and an unstated periodic BC remain byte-identical to before, in 1D, 2D and 3D, for
`no_flux`, `neumann`, `dirichlet`, `robin` and mixed alike. Only a periodic BC that has met a grid
changes.

**One 2D configuration that used to run now refuses.** A 2D periodic FP solve driven through the
tensor-diffusion route with a strong drift trips the positivity guard where it previously completed
(`Nt=40` and `Nt=160` completed before; they now raise). This is the correct direction and not a
lost capability: measured on a smooth, strictly positive, exactly periodic 2D datum under the
`N-1`-cell quadrature, the old path **created 6.3% of its own mass and did not converge away**
(6.486%, 6.324%, 6.285% at `Nt=10/40/160`), while the new one conserves to 1e-15 and closes both
seams exactly. So what used to complete was silently wrong, and what now raises is fail-loud. The
underlying 2D defect is #1835 and predates this change.
