- **`BaseNeuralSolver.get_boundary_target_values` and `BaseOptimizationSolver.get_domain_constraints` are
  deprecated** (#2512, row B2).
  Neither has a caller in the library.
  - `get_boundary_target_values` returns `None` for every `BoundaryConditions`, because it reads attributes that
    class does not have. Until it is removed, the path that does read a value goes through the owner, so a
    callable is `g(t, x)`, no longer `(point, time)`.
  - `get_domain_constraints` copies each segment's raw value outward, unread.

  Read a segment's value with `mfgarchon.geometry.boundary.types.segment_value`.
