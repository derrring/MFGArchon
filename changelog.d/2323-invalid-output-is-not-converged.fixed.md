- **`FixedPointIterator` does not report convergence when the returned solution fails output validation**
  (Issue #2323, the reporting half).
  - **Before:** `validate_solver_output` already judged a non-finite array, or a density below its -1e-10
    tolerance, as invalid. The iterator logged that at WARNING and kept `converged=True`. At 5b97f465 a 2-D
    FVM_MUSCL solve (n=17, T=0.5, sigma=0.2) returned `converged=True` with density -3.603e-10 at 20 nodes.
  - **Now:** the verdict is `converged=False`, with reason `output_invalid: ...` naming the issues, under the
    ruling recorded on #1878.
  - **Not changed:** the scheme half, making the 2-D MUSCL sub-step respect the summed bound.
