- **FP-FDM solves the Itô Fokker-Planck equation for a volatility that varies in space** (#2841). For
  `dX = alpha dt + Sigma dW` the FP equation is `d_t m + div(alpha m) = d_i d_j(A_ij m)`, `A = Sigma Sigma^T / 2`,
  which is `Laplacian(D m)` for a field (user ruling 2026-10-30; CONVENTIONS § 5). FP-FDM's implicit step, the default
  pair's, had assembled `D(x) Laplacian(m)`, so under NO_FLUX a field `volatility_kind="field"` gained mass: 7.96% by
  T = 2 on a 1-D row with sigma = 0.3 + 0.2x, and 6.96% returned by `problem.solve()` on that row, without a warning.
  The explicit step for a callable drift and `solve_fp_step_adjoint_mode` had assembled `div(D grad m)` (#1183) under
  NO_FLUX, and on a periodic grid a diffusion with D = 1 whatever sigma was; the tensor step had `div(A grad m)` for a
  per-point A. Those conserved mass, but a field tended to a constant density, where the Itô equation tends to `1/D`,
  as the particle FP does. All four now read one owner, `ito_diffusion`, which applies each step's own
  unit stencil to `A_ij m`. A scalar volatility, a uniform array and a constant tensor are unchanged.
  `FluxDiagnostics.compute_boundary_flux` reports `alpha m - grad(D m)` for a varying D, where it used D's mean in 2-D
  and n-D and the wall node's D times `grad m` in 1-D.
