- **The 1-D HJB FD-fallback Jacobian includes the Hamiltonian half of the periodic wrap entries**
  (Issue #1834). `compute_hjb_jacobian`'s per-point fallback is the default `HJBFDMSolver` path. It took
  row i's neighbours as `(i -/+ 1) % Nx`. On the endpoint-inclusive periodic grid a `TensorProductGrid`
  builds, row 0's ghost is `U[Nx-2]` and row Nx-1's is `U[1]`, so both wrap entries carried only their
  diffusion half.
  - **Symptom:** on #1822's fixture at 6c0610d2, `J[0, 19]` was -18 against a residual derivative of -155.7,
    and Newton stalled at t_idx 9, leaving a seam of 5.3e-01.
  - **Fix:** the fallback now reads its neighbours from the residual's own Laplacian bands, as the analytic
    path already did. The assembled Jacobian matches the central difference of the residual to 1e-6
    relative, for both assembly paths and both schemes, on periodic, `bc=None` and no-flux grids.
  - **Result:** the periodic seam is 7.6e-16 at Nx=21. `HJBFDMSolver` leaves #1822's list of solvers that
    do not honour PERIODIC.
  - **Also fixed:** under `bc=None`, whose residual is the legacy exclusive-periodic one, the wrap entries
    were dropped too, off by 34% relative on the central scheme.
  - **Unchanged:** non-periodic boundary conditions give byte-identical Jacobians.
