- **Two tests named after convergence rates that could not measure one** (Issue #1442 follow-up).
  `test_centered_fdm_higher_order` ran a single grid and asserted `converged or iterations >= 30`
  against a 50-iteration budget -- true either way -- on a configuration byte-identical to
  `test_centered_fdm_may_oscillate`, which asserts the same problem MUST raise (Issue #1671).
  `test_upwind_first_order_convergence` compared `result.max_error`, the final Picard residual
  rather than a discretization error, behind a `< 10.0` escape and inside a band admitting 50%
  error growth. Neither could fail for the reason its name claims, so the coverage delta is zero --
  but upwind's spatial order is now demonstrably uncovered rather than covered elsewhere, tracked
  in #1728.
