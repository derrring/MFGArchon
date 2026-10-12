- **A time default that reaches a boundary value through a chain of calls, or that an operator stores, is `None`, not
  0.0, so a value that declares `t` is refused when no time is given, not read at t = 0** (#2512, row B2). A
  constant, or `g(x)`, needs no time and reads as before.
  - The public functions are:
    - `laplacian_with_bc` and `weighted_laplacian_with_bc`, `pad_array_with_ghosts`, `get_ghost_values_nd`;
    - `tensor_calculus.gradient`, `divergence`, `laplacian`, `hessian`, `diffusion`, `tensor_diffusion` and
      `advection`;
    - `FDMApplicator.apply`, `ImplicitApplicator.apply`, `InterpolationApplicator.enforce_values`, `apply_bc` and
      `BaseNumericalSolver.apply_boundary_conditions`;
    - `refuse_unplaceable_segments`, `HJBFDMSolver.build_advection_matrix`, `apply_diffusion`, and
      `TensorProductGrid.get_gradient_operator`.
  - The operators are `LaplacianOperator`, `DiffusionOperator`, `AdvectionOperator`, `DivergenceOperator`,
    `GradientOperator`, `PartialDerivOperator`, `DirectDerivOperator` and `NormalDerivOperator`
    (`__init__` and `from_sdf`). An operator built with no time refuses such a value when it is first applied;
    `get_laplacian_operator`, `get_divergence_operator` and `get_advection_operator` take no time and build one
    so. Set its `time`, or pass `time=`, to read at a time.
  - Measured at 375122e5: `laplacian_with_bc` and `LaplacianOperator`, given a Dirichlet `g(t) = 0.25 + t` and no
    time, read g(0) in silence. They now refuse it, and with `time=0.5` give what the constant 0.75 gives.
- **Twelve private HJB-FDM and base-HJB helpers take their time as a required keyword** (#2512, row B2): every
  caller passes it. They are `_one_sided_differences_1d`, `_compute_gradient_array_1d` and `_compute_laplacian_1d`,
  and on `HJBFDMSolver` `_solve_single_timestep`, `_compute_gradients_nd`, `_one_sided_differences_nd`,
  `_evaluate_hamiltonian_nd`, `_evaluate_hamiltonian_vectorized`, `_build_advection_matrix_1d` and `_nd`, and
  `_build_linearized_operator_1d` and `_nd`.
- **Five public HJB functions keep a numeric default**, because their time is also the Hamiltonian's (#2612, which
  deprecates omitting it): `compute_hjb_residual`, `compute_hjb_jacobian`, `newton_hjb_step`,
  `solve_hjb_timestep_newton` and `HJBFDMSolver.build_linearized_operator`.
- **The implicit applicator, the n-D ghost values and the FDM ghost check now say to pass `time=`** when they refuse
  a value that declares `t`. They had advised writing it as `g(x)`.
