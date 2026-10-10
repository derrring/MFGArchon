- **BREAKING. `BlockIterator(adjoint_mode="transpose")` and `adjoint_mode="auto"` are removed.** Both now
  raise a `ValueError` that names `"jacobian_transpose"`.
  - **They were deprecated without redirecting.** The deprecation (#476, shipped in v0.17.14; its warning
    said "since v0.17.13") pointed to `"jacobian_transpose"` and named v1.0.0, but the deprecated modes kept
    running their own operator. They built the FP operator by transposing `HJBFDMSolver.build_advection_matrix`,
    and that transpose is not the HJB linearisation's.
  - **Measured** on `test_adjoint_verify_runs_2338`'s `_problem(n=40)` (no-flux, Nt = 8, T = 0.4, σ = 0.05)
    under the default `engquist_osher` Hamiltonian, at the first strict-adjoint sweep's U, over the interior
    window, against the FP solver's own operator `B = FPFDMSolver.build_advection_operator(U)`:
    - the deprecated modes' operator: max |A^T − B| from 45.2 at the first step to 245.4 at the last;
    - `build_linearized_operator`'s transpose, which `"jacobian_transpose"` uses: max |J^T − B| = 2.8e-14.
      That match is `engquist_osher`'s: under `rouy_tourin` it holds on the first sweep only (38.8 on the second).

    So "old API calls new internally, zero behaviour difference" never held.
  - **Removed earlier than the v1.0.0 the warning named.** The policy allows removal after 3 minor
    versions, and 0.22 is 5 past 0.17.14, where the deprecation shipped.
- **BREAKING. `mfgarchon.alg.numerical.adjoint.build_bc_aware_adjoint_matrix` is removed**, with the
  private helpers only it used.
  - Its only caller was `"auto"`.
  - Its docstring stated the retired reading of a reflecting FP wall as `∂m/∂n = 0`.
  - #2584 names it as a model for GFDM boundary rows; it is in git history at `25325675`.
- `BlockIterator._get_boundary_bc_types`, which read the geometry's BC raw for `"auto"`, goes with it. That
  was one of the two readers outside row B3's guard (#2512). The other is the #2529 fix.
- **`BlockIterator`'s adjoint construction gate asks for what `"jacobian_transpose"` uses.**
  - **The protocols:** `LinearizedOperatorCapable` for the HJB and `AdjointCapableFPSolver` for the FP. It used to ask, through the deprecated `validate_adjoint_capability`, for `build_advection_matrix`, which only the removed modes consumed.
  - **Admitted now:** an HJB solver with the linearisation and no `build_advection_matrix`.
  - **Refused at construction, not at the first step:** an HJB solver without `build_linearized_operator`, with the same message as before.
  - **Changed exception, which callers catch:** an FP solver lacking `solve_fp_step_adjoint_mode` now raises `NotImplementedError` at construction (was `TypeError`), like every other capability refusal on this path.
  - `validate_adjoint_capability`, the `AdjointCapableHJBSolver` export and `HJBFDMSolver.build_advection_matrix` stay. They are deprecated public names with callers, and the deprecation policy removes them (ledger #2573).
