`ImplicitApplicator` applied every Dirichlet boundary condition as `0.0` and ignored all Robin
coefficients: it read `getattr(bc, "value", 0.0)` and `getattr(bc, "alpha"/"beta", ...)` from
`BoundaryConditions`, which carries none of them — the values live on `BCSegment`. #1558 established
the owners on the parent `MeshfreeApplicator`; this subclass kept the pre-fix code.

The defect was unreachable in practice for a second reason, now also fixed: `_detect_boundary_points`
called `is_on_boundary(points, tolerance=...)`, while `ImplicitDomain` names that parameter `tol` and
the whole implicit family inherits it, so `apply()` raised `TypeError` for every real geometry. The
call is now positional. (Two different sevens, worth keeping apart: seven classes *define* `is_on_boundary`, six of them
naming the parameter `tolerance`; and seven classes *resolve* to `tol` -- `ImplicitDomain` plus the
six that inherit from it. Only one definition needs renaming.) The suite missed both because its fixture supplied a geometry written against the
applicator's call rather than against `GeometryProtocol`.
