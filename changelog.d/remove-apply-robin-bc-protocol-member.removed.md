- **`AdvancedBoundaryHandler.apply_robin_bc` removed from the boundary protocol.** It had zero
  implementations, and the signature is why: `apply_robin_bc(values, alpha, beta, gamma) -> values`
  says "hand me a solution and I will return one satisfying Robin", but a Robin condition
  constrains the *operator* — the ghost value that closes a stencil, or the boundary terms of a
  weak form — and cannot in general be recovered by editing a solution vector. The three owners
  that do impose it each need more than `values`: `RobinCalculator.compute` and `ghost_cell_robin`
  take `(interior_value, dx, side, alpha, beta, rhs)` and return a ghost value, and the weak-form
  path adds terms to the bilinear form. None is expressible as `values -> values`. The protocol
  keeps `get_boundary_normals`, which has three real implementations.
