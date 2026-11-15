- **`mfgarchon.utils` and `mfgarchon.utils.numerical` re-export nothing from a higher layer** (#2938, stage 2a). A
  package `__init__` re-exports names of its own layer or below. Import each name below from its owner:
  - from `mfgarchon.utils`:
    - `sdf_box`, `sdf_complement`, `sdf_difference`, `sdf_gradient`, `sdf_intersection`, `sdf_smooth_intersection`,
      `sdf_smooth_union`, `sdf_sphere`, `sdf_union`: from `mfgarchon.geometry.implicit.sdf_utils`;
    - `Hyperrectangle`, `Hypersphere`, `ImplicitDomain`: from `mfgarchon.geometry.implicit`;
    - `BoxObstacle`, `CircleObstacle`, `Complement`, `Difference`, `Intersection`, `RectangleObstacle`, `SphereObstacle`,
      `Union`, `create_box_obstacle`, `create_circle_obstacle`, `create_rectangle_obstacle`, `create_sphere_obstacle`:
      from `mfgarchon.utils.geometry`, which defines them;
    - `composite_drift`, `density_dependent_drift`, `prescribed_drift`, `zero_drift`: from
      `mfgarchon.utils.drift_helpers`;
    - `CoefficientField`, `check_adi_compatibility`, `get_spatial_grid`: from `mfgarchon.utils.pde_coefficients`;
    - `DualityStatus`, `DualityValidationResult`, `check_solver_duality`, `validate_scheme_config`: from
      `mfgarchon.utils.adjoint_validation`;
    - `load_experiment_data`, `load_experiments_from_dir`, `plot_comparison_final_m`, `plot_comparison_total_mass`,
      `save_experiment_data`: from `mfgarchon.utils.experiment_manager`;
    - `SparseMatrixBuilder`, `SparseSolver`, `estimate_sparsity`, `sparse_matmul`: from
      `mfgarchon.utils.sparse_operations`, which builds Laplacians on a tensor grid and now sits in the operators
      layer;
    - `cli_main`: `main` in `mfgarchon.utils.cli`.
  - from `mfgarchon.utils.numerical`:
    - the nine `sdf_*` functions above: from `mfgarchon.geometry.implicit.sdf_utils`;
    - `MeshDistances`, `compute_distances_for_eoc_study`, `compute_mesh_distances`: from
      `mfgarchon.geometry.meshes.mesh_distances`;
    - `BoundaryFluxResult`, `FluxDiagnostics`, `FluxSummary`, `compute_mass_conservation_error`: from
      `mfgarchon.utils.numerical.flux_diagnostics`;
    - `HJBPolicyProblem`, `create_lq_policy_problem`, `policy_iteration_hjb`: from
      `mfgarchon.utils.numerical.hjb_policy_iteration`.
- **A bare `import mfgarchon` no longer loads** `mfgarchon.utils.sparse_operations`, `mfgarchon.utils.geometry`,
  `mfgarchon.utils.cli`, `mfgarchon.utils.experiment_manager`, `mfgarchon.utils.drift_helpers`,
  `mfgarchon.utils.numerical.flux_diagnostics`, `mfgarchon.utils.numerical.hjb_policy_iteration`,
  `mfgarchon.geometry.implicit.sdf_utils` or `mfgarchon.geometry.meshes.mesh_distances`. Code that reached one of them
  as an attribute after `import mfgarchon` imports it explicitly instead, for example `import mfgarchon.utils.geometry`.
- **`sample_from_scattered_density` moves to `mfgarchon.utils.particle_utils`.** It is the one sampler that knows a
  periodic domain, and periodicity is geometry's. `mfgarchon.utils.numerical.particle.sample_from_scattered_density`
  and `mfgarchon.utils.numerical.particle.sampling.sample_from_scattered_density` are removed. Its debug log now comes
  from the `mfgarchon.utils.particle_utils` logger.
