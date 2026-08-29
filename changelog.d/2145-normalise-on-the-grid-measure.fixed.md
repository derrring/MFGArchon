- **The initial density is normalised on the geometry's own measure** (Issue #2145). The constructor
  rescaled `m_initial` so that `sum(m) * dx` came out 1, while `TensorProductGrid` is
  endpoint-inclusive — the wall lies ON `x_0`, the two end nodes own half a cell each, and the
  measure is the trapezoid. Normalising with one functional while the FP wall conserves the other is
  how a solve reports perfect conservation of a quantity nobody asked about. A raw Gaussian now
  arrives with `grid.integrate(m) == 1` exactly and `sum(m)*dx == 1.0075`; before, the two were the
  other way round. `problem.initial_mass_measure` names which measure produced the number, because
  these branches are different objects rather than fallbacks of one — a network has no cell volume
  and an unstructured geometry has no quadrature.
