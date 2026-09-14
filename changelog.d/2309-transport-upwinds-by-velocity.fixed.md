- **`AdvectionOperator(scheme="upwind")` upwinds by the sign of the velocity** (Issue #2309). Its
  non-conservative path (the default, `mass_conservative=False`) took each one-sided difference by the
  sign of the differentiated field. For transport that is non-monotone: a minimum explicit update
  coefficient of -CFL, and blow-up at constant velocity. It now uses `gradient_upwind_by_velocity`
  (gradient form) and `divergence_upwind_by_velocity` (divergence form), and every update coefficient
  is nonnegative. `tensor_calculus.advection` calls the same two functions, byte-identical on its
  divergence form. Production FP solvers pass `mass_conservative=True` and do not change.
