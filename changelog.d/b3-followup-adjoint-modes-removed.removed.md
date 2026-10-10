- **BREAKING. `BlockIterator(adjoint_mode="transpose")` and `adjoint_mode="auto"` are removed.** Both now
  raise a `ValueError` that names `"jacobian_transpose"`.
  - **The deprecation was misleading.** They were deprecated in v0.17.13 as aliases to be removed in v1.0.0,
    but they ran a different operator. They built the FP operator by transposing
    `HJBFDMSolver.build_advection_matrix`, and that transpose is not the HJB linearisation's.
  - **Measured** on a 40-point no-flux grid, over the interior window, against the FP solver's own operator
    `B = FPFDMSolver.build_advection_operator(U)`:
    - the deprecated modes' operator: max |A^T − B| = 245;
    - `build_linearized_operator`'s transpose, which `"jacobian_transpose"` uses: max |J^T − B| = 2.8e-14.

    So "old API calls new internally, zero behaviour difference" never held.
  - **Removed earlier than the v1.0.0 the warning named.** The policy allows removal after 3 minor
    versions, and 0.22 is 5 past 0.17.13.
- **BREAKING. `mfgarchon.alg.numerical.adjoint.build_bc_aware_adjoint_matrix` is removed**, with the
  private helpers only it used.
  - Its only caller was `"auto"`.
  - Its docstring stated the retired reading of a reflecting FP wall as `∂m/∂n = 0`.
  - #2584 names it as a model for GFDM boundary rows; it is in git history at `25325675`.
- `BlockIterator._get_boundary_bc_types`, which read the geometry's BC raw for `"auto"`, goes with it. That
  was one of the two readers outside row B3's guard (#2512). The other is the #2529 fix.
