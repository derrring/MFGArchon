- **`tensor_calculus` no longer carries private copies of the finite-difference stencils** (Issue
  #1794). `_gradient_central`, `_gradient_forward`, `_gradient_backward`, `_gradient_upwind` and
  `_fix_boundaries_one_sided` are gone; `tensor_calculus.gradient` calls
  `operators/stencils/finite_difference` directly. The per-point legacy branch of
  `base_hjb._calculate_derivatives`, a third statement of the upwind rule, calls it too.
