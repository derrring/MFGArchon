- **`AdvectionOperator(scheme="upwind")` upwinds by the sign of the velocity** (Issue #2309). Its
  non-conservative path (the default, `mass_conservative=False`) took each one-sided difference by the
  sign of the differentiated field. For transport that is non-monotone: a minimum explicit update
  coefficient of -CFL, and blow-up at constant velocity. The gradient form now calls
  `gradient_upwind_by_velocity`. The divergence form calls `divergence_upwind_by_velocity`, which splits
  each node's flux by the sign of that node's velocity, so the explicit update has nonnegative
  coefficients for any velocity field. `tensor_calculus.advection` calls the same two functions. Its
  divergence form used to select a whole node flux by the sign of the face-averaged velocity, which
  sends mass downwind where the flow diverges: a coefficient of -CFL at a velocity step. Both pad the
  velocity by holding its edge value at a wall rather than with the density's boundary condition, so
  a Dirichlet value no longer reads as a flow direction. Production FP solvers pass
  `mass_conservative=True` and do not change.
