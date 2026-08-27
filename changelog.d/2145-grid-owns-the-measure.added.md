- **`TensorProductGrid` now owns the measure**: `quadrature_weights(axis)` gives the control volume
  each node holds and `integrate(field)` reduces a field with them (Issue #2145). The grid is
  endpoint-inclusive, so the two end nodes own **half a cell** each — those are the trapezoid
  weights, and `sum(m)*dx` is a different functional that over-counts the ends by
  `dx*(m[0]+m[-1])/2`, 3.5% on a standard fixture before anything evolves. Written on coordinates,
  so a graded grid is not a special case; `integrate` reduces the trailing axes, so a
  `(time, *spatial)` history gives one value per row, and a corner owns the product of half-cells.
  `mass_drift` accepts a grid and works in n-D through it, where it previously raised on any 2-D
  field.
