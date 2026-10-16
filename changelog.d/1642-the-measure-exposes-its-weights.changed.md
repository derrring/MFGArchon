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
    they computed `sum(f) * cell` and `sum(f) / n`. For a float64 field these round differently in the last bits;
    `"node-sum"` gives the same bytes.
  - `integrate_on_weights` sums in float64. So every non-grid branch now returns float64 for a float32 or integer
    field, where it summed in, and returned, the field's own dtype (`"node-sum"`) or summed in it (the other two):
    a float32 field's integral moves by about 1e-7 relative, the float32 sum's rounding.
  - A field whose trailing axes are not shaped like the weights is refused there, rather than summed.
  - `FPParticleSolver` reads its mass weights from the measure. It no longer rebuilds them from the per-axis
    quadrature weights.
  - `FPFVMSolver`'s control volumes are `TensorProductGrid.measure_weights()`. Through `integrate_on_weights` now:
    `FluxDiagnostics`' 1-D and 2-D total mass and face fluxes (its n-D path is still the rectangle rule, #2274),
    `ConvergenceWrapper`'s mass change, `validate_mfg_solution`'s total mass, and `mass_fabricated_by_clip`'s
    negative and positive mass on given weights. Each is the same integral as before, summed in another order, so
    its value can move in the last bits.
