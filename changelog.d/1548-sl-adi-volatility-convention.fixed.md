- **`adi_diffusion_step` now reads a `(d,d)` sigma as the SDE volatility, matching the package
  convention** (Issue #1548). It documented and treated a full-tensor `sigma` as the *covariance*
  (`sigma@sigma^T`), while the package convention — and the caller (`problem.sigma`), and this
  function's own scalar/1-D branches — is that `sigma` is the volatility `Sigma` with `D = Sigma@Sigma^T/2`.
  A symmetric volatility `Sigma` (e.g. `0.3*I`) was silently over-diffused (`D = Sigma/2 = 0.15`
  instead of `0.045`, ~3.3x), and a Cholesky-factor `Sigma` (lower-triangular) tripped a symmetry
  `ValueError`. The `(d,d)` branch now forms the covariance `C = Sigma@Sigma^T` internally and drives
  both the diagonal ADI (`D_d = C[d,d]/2`) and the explicit cross-derivative (`C[i,j]`) from it, so an
  isotropic `Sigma = c*I` is byte-identical to the scalar-`c` path; the symmetry precondition on the
  input is removed (`C` is symmetric PSD by construction). Scalar and 1-D-diagonal sigma are
  unaffected. Fixes the dual-convention seam shared by HJB-SL and FP-SL-adjoint (both call this).
