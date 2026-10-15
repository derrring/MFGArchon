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
  - A field whose trailing axes are not shaped like the weights is refused there, rather than summed.
  - `FPParticleSolver` reads its mass weights from the measure. It no longer rebuilds them from the per-axis
    quadrature weights.
