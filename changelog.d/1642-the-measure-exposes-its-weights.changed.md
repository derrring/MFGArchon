- **`MFGProblem.spatial_measure()` returns the measure's per-node weights** (#1642, stage 0 of the
  `coupling_cost` wiring).
  - `SpatialMeasure` now holds `weights` and `name`. Its `integrate` is `sum(f * weights)` over the trailing axes in
    every branch, through one helper, `utils.numerical.quadrature.integrate_on_weights`, which
    `TensorProductGrid.integrate` uses too.
  - The `"grid"` branch reads the new `TensorProductGrid.measure_weights()`.
  - A geometry that has an `integrate` but no `measure_weights` is now refused by name. Its weights cannot be
    recovered from `integrate` short of integrating one indicator field per node. No geometry in the library is
    affected: `TensorProductGrid` is the only one with an `integrate`.
  - The `"uniform-cell"` and `"point-average"` branches now compute `sum(f * cell)` and `sum(f * (1/n))`, where
    they computed `sum(f) * cell` and `sum(f) / n`. These round differently in the last bits.
  - The `"node-sum"` branch gives the same bytes for a float64 field. A float32 or integer field is now summed in
    float64 and returns float64, where it was summed and returned in its own dtype.
  - A field whose trailing axes are not shaped like the weights is refused there, rather than summed.
  - `FPParticleSolver` reads its mass weights from the measure. It no longer rebuilds them from the per-axis
    quadrature weights.
  - `FPFVMSolver`'s control volumes are `TensorProductGrid.measure_weights()`. `FluxDiagnostics`' total mass and
    face fluxes, `ConvergenceWrapper`'s mass change and `validate_mfg_solution`'s total mass sum through
    `integrate_on_weights`.
