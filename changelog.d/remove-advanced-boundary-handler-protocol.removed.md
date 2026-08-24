- **BREAKING.** `AdvancedBoundaryHandler` removed from `mfgarchon.geometry.boundary`. Completes the
  removal begun by `remove-apply-robin-bc-protocol-member`: with `apply_robin_bc` gone the protocol
  added exactly one member over `BoundaryHandler` — `get_boundary_normals` — and **nothing
  implemented it.** Measured by sweeping the package: 371 modules, 1526 classes, zero satisfying it,
  zero `isinstance`/`issubclass` sites, zero call sites; its only reachable surface was a lazy
  export. The one member it added has no agreed signature either — `CollocationPointSet` defines
  `get_boundary_normals(self)`, `MeshfreeApplicator` `(self, points)` and `Hyperrectangle`
  `(self, points, eps)` — so the protocol described one of three shapes and matched none of the
  classes. Being `@runtime_checkable`, which compares names and not signatures, it was also a latent
  wrong positive: `Hyperrectangle` already matched three of its four members and was one attribute
  away from silently classifying as a solver boundary handler. `BoundaryHandler` itself is
  unaffected and keeps its single implementer, `HJBSemiLagrangianSolver`.
