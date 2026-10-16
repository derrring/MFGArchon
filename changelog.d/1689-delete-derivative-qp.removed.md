- **The derivative-coefficient QP is removed** (#1689, members 2 and 6). It constrained the solution's Taylor
  coefficients (a non-positive Laplacian estimate, a gradient bounded by it), not the stencil's weights, so it was not
  a monotone scheme. On its default OSQP branch the constraints were never encoded, and the trigger that decided where
  it ran read u's derivatives, not the stencil.
  - `HJBGFDMSolver(monotonicity_scheme="qp_m_matrix")` and the applications `"adaptive"` and `"always"`.
    `monotonicity_scheme` is `"none"` or `"joint_socp"`, and `monotonicity_application` is `"precompute"` or None.
  - `HJBGFDMSolver`'s `qp_solver`, `qp_warm_start` and `qp_constraint_mode` keywords, `compute_all_derivatives(use_qp=)`
    and `solver.qp_stats`.
  - `MonotonicityEnforcer`. Its module keeps `verify_assembled_m_matrix` and `critical_drift_for_dmp`.
  - `QPSolver` and `QPCache` (`mfgarchon.utils`, `mfgarchon.utils.numerical.qp_utils`), with their guide section,
    example and benchmark: nothing in the library calls them, and their OSQP branch dropped the constraints it was
    given. For a constrained least-squares problem use cvxpy or `scipy.optimize.minimize`.
  - `GFDMConfig.qp` and `QPConfig`, replaced by `GFDMConfig.monotonicity_scheme` (`"none"` or `"joint_socp"`). The
    two places that read the QP level disagreed: `HJBGFDMSolver.from_config` sent `"always"` to `qp_m_matrix` and
    the config translator sent it to `joint_socp`.
  - #1991's per-point-path test, which ran on `qp_m_matrix`, runs on `joint_socp` and is no longer marked slow.
