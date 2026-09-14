- **Non-HJB consumers of `gradient_upwind` change at discrete local minima** (Issue #2308).
  `AdvectionOperator(scheme="upwind")`'s non-conservative path and `reinitialize(method="pde")`
  select a one-sided difference by the sign of the field, which is the wrong concept for each:
  transport upwinds by velocity, reinitialisation by the sign of the initial level set. The rule change
  moves both at minima, making reinitialisation worse where the initial level set is negative. Neither
  is a default path, and neither has a production caller; tracked in #2309 and #2310. Also moved at
  minima: `DirectDerivOperator(scheme="upwind")`, `tensor_calculus.advection` on its default
  `form="gradient", method="upwind"`, and the velocity in `HJBFDMSolver.build_advection_matrix`, reached
  only through the deprecated `transpose`/`auto` adjoint modes.
