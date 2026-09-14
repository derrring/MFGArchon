- **`AdvectionOperator(scheme="upwind")` upwinds by the sign of the velocity** (Issue #2309). Its
  non-conservative path (the default, `mass_conservative=False`) took each one-sided difference by the
  sign of the differentiated field. For transport that is non-monotone: a minimum explicit update
  coefficient of -CFL, and blow-up at constant velocity.
  - The gradient form now calls `gradient_upwind_by_velocity`.
  - The divergence form calls `divergence_upwind_by_velocity`, a face-velocity donor cell:
    `F_{i+1/2} = max(v_{i+1/2}, 0) m_i + min(v_{i+1/2}, 0) m_{i+1}`. It is first-order consistent
    where the velocity changes sign. It is conservative on a torus. Its explicit update has
    nonnegative coefficients for `2 max|v| dt / h <= 1` in the interior, on a torus and at no-flux
    walls.
  - `tensor_calculus.advection` calls the same two functions. Its divergence form used to select a
    whole node flux by the sign of the face velocity, which sends mass downwind where the flow
    diverges: a coefficient of -CFL at a velocity step.
  - Under `scheme="upwind"` both pad the velocity with its edge value at a wall, not with the
    density's boundary condition, so a Dirichlet value no longer reads as a flow direction.
  - `scheme="centered"` and the production FP solvers, which pass `mass_conservative=True`, do not
    change.
