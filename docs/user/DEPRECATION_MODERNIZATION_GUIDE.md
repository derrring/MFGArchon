# Deprecation Modernization Guide

**Auto-generated** by `scripts/generate_deprecation_guide.py`
**Total deprecated items**: 72
**Versions covered**: v0.21.0, v0.20.5, v0.20.0, v0.19.2, v0.19.0, v0.18.7, v0.18.6, v0.18.0, v0.17.6, v0.17.1, v0.17.0, v0.16.11, v0.12.0

---

## Overview

This guide documents deprecated usage patterns in MFGArchon and provides
migration paths to modern APIs. All deprecated patterns emit warnings at
runtime and will be removed at the version specified.

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
| `FPSLAdjointSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `FPSLJacobianSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `FPSLSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `MeshlessGalerkinFPSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `NetworkFPSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |
| `WeakFormFPSolver.solve_fp_system()` | itself deprecated | `drift_field` -> `potential_field` |

---

## Deprecated since v0.21.0

*2 items*

### Parameters

- **`drift_field`** in `FPNetworkSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `NetworkFPSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]

---

## Deprecated since v0.20.5

*2 items*

### Parameters

- **`diffusion`** in `AdjointConsistentProvider.__init__()` — use `sigma` instead (remove by v0.25.0)

### Functions / Classes

- **`HJBWenoSolver()`** — use `HJBWENOSolver` instead (remove by v1.0.0)

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

- **`optimal_control_drift()`** — use `use H.optimal_control(x, m, grad_U, t) directly, or let FixedPointIterator handle it automatically` instead (remove by v0.25.0)

---

## Deprecated since v0.18.7

*1 items*

### Parameters

- **`tensor_volatility_field`** in `HJBFDMSolver.solve_hjb_system()` — use `volatility_field (pass (d,d) array or callable returning (d,d))` instead (remove by v0.25.0)

---

## Deprecated since v0.18.6

*4 items*

### Parameters

- **`velocity_field`** in `FPFDMSolver.solve_fp_system()` — use `drift_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `FPSLAdjointSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `FPSLJacobianSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]
- **`drift_field`** in `FPSLSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0) [see *Do not migrate these across solvers*: `drift_field`]

---

## Deprecated since v0.18.0

*4 items*

### Functions / Classes

- **`GradientComponentOperator()`** — use `PartialDerivOperator` instead (remove by v1.0.0)
- **`_compute_sdf_gradient()`** — use `use mfgarchon.operators.differential.function_gradient() instead` instead (remove by v0.25.0)
- **`_compute_upwind_advection()`** — use `AdvectionOperator` instead (remove by v0.25.0)
- **`mixed_bc()`** — use `Use BoundaryConditions(segments=[...]) directly` instead (remove by v0.25.0)

---

## Deprecated since v0.17.6

*2 items*

### Functions / Classes

- **`FPSLAdjointSolver()`** — use `FPSLSolver` instead (remove by v1.0.0)
- **`__init__()`** — use `FPSLSolver` instead (remove by v0.25.0)

---

## Deprecated since v0.17.1

*3 items*

### Functions / Classes

- **`MFGDriftField()`** — use `DriftField` instead (remove by v1.0.0)
- **`_solve_fp_1d()`** — use `solve_fp_system` instead (remove by v0.25.0)
- **`_solve_fp_1d_with_callable()`** — use `solve_fp_system` instead (remove by v0.25.0)

---

## Deprecated since v0.17.0

*39 items*

### Parameters

- **`enable_curriculum`** in `AdaptiveTrainingConfig.__init__()` — use `training_mode` instead (remove by v0.25.0)
- **`enable_multiscale`** in `AdaptiveTrainingConfig.__init__()` — use `training_mode` instead (remove by v0.25.0)
- **`enable_refinement`** in `AdaptiveTrainingConfig.__init__()` — use `training_mode` instead (remove by v0.25.0)
- **`use_control_variates`** in `DGMConfig.__init__()` — use `variance_reduction` instead (remove by v0.25.0)
- **`use_importance_sampling`** in `DGMConfig.__init__()` — use `variance_reduction` instead (remove by v0.25.0)
- **`use_batch_norm`** in `DeepONetConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`use_layer_norm`** in `DeepONetConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`tensor_diffusion_field`** in `FPFDMSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`volatility_matrix`** in `FPFDMSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPNetworkSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh1D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh1D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh2D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh2D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh3D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh3D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`m_initial_condition`** in `NetworkFPSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`use_batch_norm`** in `PINNConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`use_layer_norm`** in `PINNConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`dimension`** in `TensorProductGrid.__init__()` — use `len(bounds) (dimension is inferred from bounds)` instead (remove by v0.25.0)
- **`num_points`** in `TensorProductGrid.__init__()` — use `Nx_points` instead (remove by v0.25.0)
- **`show_edges`** in `UnstructuredMesh.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `UnstructuredMesh.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `_MeshGeneratorBase.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `_MeshGeneratorBase.visualize_mesh()` — use `mode` instead (remove by v0.25.0)

### Functions / Classes

- **`__init__()`** — use `Use TaylorOperator from gfdm_strategies instead: from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import TaylorOperator` instead (remove by v0.25.0)
- **`_deprecated_xp_zeros()`** — use `Use backend.zeros() instead for device consistency.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_1d()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_2d()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_3d()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_nd()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`compute_adjoint_consistent_bc_values()`** — use `Use mfgarchon.alg.numerical.adjoint.compute_adjoint_consistent_bc_values instead.` instead (remove by v0.25.0)
- **`compute_boundary_log_density_gradient_1d()`** — use `Use mfgarchon.alg.numerical.adjoint.compute_boundary_log_density_gradient_1d instead.` instead (remove by v0.25.0)
- **`create_adjoint_consistent_bc_1d()`** — use `Use mfgarchon.alg.numerical.adjoint.create_adjoint_consistent_bc_1d instead.` instead (remove by v0.25.0)
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
