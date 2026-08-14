`ImplicitApplicator` applied every Dirichlet boundary condition as `0.0` and ignored all Robin
coefficients: it read `getattr(bc, "value", 0.0)` and `getattr(bc, "alpha"/"beta", ...)` from
`BoundaryConditions`, which carries none of them — the values live on `BCSegment`. #1558 established
the owners on the parent `MeshfreeApplicator`; this subclass kept the pre-fix code.

The defect was unreachable in practice for a second reason, now also fixed: `_detect_boundary_points`
called `is_on_boundary(points, tolerance=...)`, but all seven `ImplicitDomain` subclasses the package
ships name that parameter `tol`, so `apply()` raised `TypeError` for every real geometry. The call is
now positional. The suite missed both because its fixture supplied a geometry written against the
applicator's call rather than against `GeometryProtocol`.
