# Deprecation Modernization Guide

**Auto-generated** by `scripts/generate_deprecation_guide.py`
**Total deprecated items**: 104
**Versions covered**: v0.22.0, v0.21.0, v0.20.0, v0.19.2, v0.19.0, v0.18.6, v0.18.0, v0.17.6, v0.17.0, v0.16.11, v0.12.0

---

## Overview

This guide documents deprecated usage patterns in MFGArchon and provides
migration paths to modern APIs. Deprecated patterns emit warnings at
runtime and will be removed at the version specified. Refused parameters
already raise a TypeError naming the replacement; the version is when that
refusal goes.

To find deprecated usage in your code:
```bash
python -W error::DeprecationWarning -c 'import mfgarchon; ...'
```

---

## Do not migrate these across solvers

The identifiers below are **deprecated in one place and the recommended replacement in another**. That is not a mistake in this guide: the same word names different quantities on different solvers, and each row is correct for the API it names.

It does mean a migration you read on one row **does not transfer** to another solver. Both parameters usually exist on both solvers, so applying the wrong one is accepted silently and changes the answer rather than raising. Check the target solver's `solve_*` docstring for what the parameter means there before renaming anything.

### `drift_field`

| in this API | `drift_field` is | migration on that row |
|---|---|---|
| `FPFDMSolver.solve_fp_system()` | the destination | `velocity_field` -> `drift_field` |
| `FPFEMSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `FPNetworkSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `FPSLJacobianSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `FPSLSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `MeshlessGalerkinFPSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `NetworkFPSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `WeakFormFPSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |

---

## Deprecated since v0.22.0

*61 items*

### Parameters

- **`m_initial`** in `FPFEMSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPGFDMSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`m_initial`** in `MeshlessGalerkinFPSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`m_initial`** in `WeakFormFPSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)

### Functions / Classes

- **`HamiltonianAdapter()`** — use `bind_user_callable(func, HAMILTONIAN_SLOTS, role=...) from mfgarchon.types.callable_protocols, or a HamiltonianBase called by keyword` instead (remove by v0.25.0)
- **`adapt_hamiltonian()`** — use `bind_user_callable(func, HAMILTONIAN_SLOTS, role=...) from mfgarchon.types.callable_protocols, or a HamiltonianBase called by keyword` instead (remove by v0.25.0)
- **`create_hamiltonian_adapter()`** — use `bind_user_callable(func, HAMILTONIAN_SLOTS, role=...) from mfgarchon.types.callable_protocols, or a HamiltonianBase called by keyword` instead (remove by v0.25.0)

### Refused parameters (already raise TypeError)

- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `BlockIterator.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `DiffusionOperator.from_volatility()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `FPFDMSolver.solve_fp_step_adjoint_mode()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPFDMSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `FPFEMSolver.solve_fp_step_adjoint_mode()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPFEMSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPFVMSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPGFDMSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPNetworkSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPParticleSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPSLJacobianSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FPSLSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FictitiousPlayIterator.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `FixedPointIterator.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBFDMSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBFEMSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBGFDMSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBHowardSolver.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBHowardSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBNetworkSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBSemiLagrangianSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `HJBWENOSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `MFGResidual.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `MeshlessGalerkinFPSolver.solve_fp_step_adjoint_mode()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `MeshlessGalerkinFPSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `MeshlessGalerkinHJBSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `sigma_function=` in `MonotonicityEnforcer.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `NetworkFPSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `NetworkHJBSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `NetworkPolicyIterationHJBSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `NewtonMFGSolver.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `PenaltyHJBSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `VariationalMFGProblem.__init__()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `sigma=` in `WeakFormFPSolver.solve_fp_step_adjoint_mode()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `WeakFormFPSolver.solve_fp_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `WeakFormHJBSolver.solve_hjb_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0, once no longer blocked: the function takes **kwargs, which would accept a retired name and ignore it
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `_solve_fp_nd_full_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `build_diffusion_matrix()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `build_diffusion_matrix_1d()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `build_diffusion_matrix_2d()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `build_diffusion_matrix_from_geometry()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `check_adi_compatibility()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma_at_n=` in `compute_hjb_jacobian()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma_at_n=` in `compute_hjb_residual()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `create_obstacle_variational_mfg()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `create_quadratic_variational_mfg()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `diffusion_from_volatility()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=` in `diffusion_from_volatility_torch()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=`, `sigma_kind=` in `fp_source()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma=`, `sigma_kind=` in `hjb_source()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma_at_n=` in `newton_hjb_step()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `solve_fp_nd_full_system()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `diffusion_field=`, `tensor_diffusion_field=`, `volatility_field=`, `volatility_matrix=` in `solve_hjb_system_backward()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0
- `sigma_at_n=` in `solve_hjb_timestep_newton()` — refused with a TypeError that names the replacement (#2378); the refusal goes by v0.25.0

---

## Deprecated since v0.21.0

*2 items*

### Parameters

- **`drift_field`** in `FPNetworkSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `NetworkFPSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]

---

## Deprecated since v0.20.0

*3 items*

### Parameters

- **`drift_field`** in `FPFEMSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `MeshlessGalerkinFPSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `WeakFormFPSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]

---

## Deprecated since v0.19.2

*9 items*

### Parameters

- **`damping_factor`** in `BlockGaussSeidelIterator.__init__()` — use `relaxation` instead (remove by v0.25.0)
- **`damping_factor`** in `BlockIterator.__init__()` — use `relaxation` instead (remove by v0.25.0)
- **`damping_factor_M`** in `BlockIterator.__init__()` — use `relaxation_M` instead (remove by v0.25.0)
- **`damping_factor`** in `BlockJacobiIterator.__init__()` — use `relaxation` instead (remove by v0.25.0)
- **`damping_factor`** in `FixedPointSolver.__init__()` — use `relaxation` instead (remove by v0.25.0)
- **`damping_factor`** in `HJBFDMSolver.__init__()` — use `relaxation` instead (remove by v0.25.0)
- **`damping_factor`** in `MultiPopulationIterator.__init__()` — use `relaxation` instead (remove by v0.25.0)
- **`damping_factor`** in `create_network_mfg_solver()` — use `relaxation` instead (remove by v0.25.0)
- **`damping`** in `create_simple_network_solver()` — use `relaxation` instead (remove by v0.25.0)

---

## Deprecated since v0.19.0

*1 items*

### Functions / Classes

- **`optimal_control_drift()`** — use `use H.optimal_control(t=t, x=x, p=grad_U, m=m) directly, or let FixedPointIterator handle it automatically` instead (remove by v0.25.0)

---

## Deprecated since v0.18.6

*3 items*

### Parameters

- **`velocity_field`** in `FPFDMSolver.solve_fp_system()` — use `drift_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `FPSLJacobianSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `FPSLSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]

---

## Deprecated since v0.18.0

*2 items*

### Functions / Classes

- **`_compute_sdf_gradient()`** — use `use mfgarchon.operators.differential.function_gradient() instead` instead (remove by v0.25.0)
- **`mixed_bc()`** — use `Use BoundaryConditions(segments=[...]) directly` instead (remove by v0.25.0)

---

## Deprecated since v0.17.6

*1 items*

### Functions / Classes

- **`__init__()`** — use `FPSLSolver` instead (remove by v0.25.0)

---

## Deprecated since v0.17.0

*20 items*

### Parameters

- **`m_initial_condition`** in `FPNetworkSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh1D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh1D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh2D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh2D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh3D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh3D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`m_initial_condition`** in `NetworkFPSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`dimension`** in `TensorProductGrid.__init__()` — use `len(bounds) (dimension is inferred from bounds)` instead (remove by v0.25.0)
- **`num_points`** in `TensorProductGrid.__init__()` — use `Nx_points` instead (remove by v0.25.0)
- **`show_edges`** in `UnstructuredMesh.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `UnstructuredMesh.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `_MeshGeneratorBase.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `_MeshGeneratorBase.visualize_mesh()` — use `mode` instead (remove by v0.25.0)

### Functions / Classes

- **`__init__()`** — use `Use TaylorOperator from gfdm_strategies instead: from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import TaylorOperator` instead (remove by v0.25.0)
- **`create_solver()`** — use `Use the new three-mode solving API instead (Issue #580):
  - Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)
  - Expert Mode: problem.solve(hjb_solver=hjb, fp_solver=fp)
  - Auto Mode: problem.solve()
See examples/basic/three_mode_api_demo.py for details.` instead (remove by v0.25.0)
- **`get_ghost_values_nd()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`validate_adjoint_capability()`** — use `validate_scheme_pairing` instead (remove by v0.25.0)
- **`wrap_positions()`** — use `Use mfgarchon.geometry.boundary.periodic.wrap_positions instead.` instead (remove by v0.25.0)

### Properties

- **`num_points`** (property) — use `Use Nx_points instead.` instead (remove by v0.25.0)

---

## Deprecated since v0.16.11

*1 items*

### Functions / Classes

- **`__init__()`** — use `Use ZeroFluxCalculator instead for J*n = 0 (mass conservation).` instead (remove by v0.25.0)

---

## Deprecated since v0.12.0

*1 items*

### Functions / Classes

- **`apply_boundary_conditions()`** — use `Use MeshfreeApplicator from mfgarchon.geometry.boundary instead.` instead (remove by v0.25.0)

---

## Migration Help

If you encounter a deprecation warning not listed here,
please file an issue at https://github.com/derrring/MFGArchon/issues
