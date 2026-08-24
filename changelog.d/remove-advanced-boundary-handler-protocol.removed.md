- **BREAKING.** `AdvancedBoundaryHandler` removed. Completes the removal begun by
  `remove-apply-robin-bc-protocol-member`: with `apply_robin_bc` gone the protocol added exactly one
  member over `BoundaryHandler` — `get_boundary_normals` — and **nothing implemented it.** Measured
  by sweeping the package: **371 modules, 631 classes** (classes defined under `mfgarchon`, deduped;
  0 import failures), zero satisfying it, zero `isinstance`/`issubclass` sites and zero call sites
  *in this repository* — out-of-repo sites are why this is marked breaking. **Three import paths
  break, not one**: `from mfgarchon.geometry.boundary import AdvancedBoundaryHandler` (the lazy
  export), `from mfgarchon.geometry.boundary.protocols import AdvancedBoundaryHandler` (the class
  was defined at module level), and `import *` from that module, which its `__all__` advertised. The
  symbol shipped in **v0.21.0**, the latest published release. The one member it added has no agreed
  signature either — `CollocationPointSet` defines `get_boundary_normals(self)`, `MeshfreeApplicator`
  `(self, points)` and `Hyperrectangle` `(self, points, eps)` — so the protocol described one of
  three shapes and matched none of the classes. Being `@runtime_checkable`, which compares names and
  not signatures, it was also a latent wrong positive: `Hyperrectangle` already matched three of its
  four members and was one attribute away from silently classifying as a solver boundary handler.
  No redirect is offered because there is no new standard to redirect to: nothing implements or
  replaces the protocol. Per the pre-1.0 paragraph in `AGENTS.md`, this ships in the current
  `v0.MINOR.x` line via this fragment — the same treatment `apply_robin_bc` had one commit earlier.
- `BoundaryHandler` itself is **not** removed here, but "unaffected" would overstate it: it keeps a
  single implementer (`HJBSemiLagrangianSolver`) out of 631 classes, its only consumer
  `validate_boundary_handler` is called nowhere outside an `if __name__ == "__main__"` block, and
  `Hyperrectangle` matches two of its three members with incompatible signatures — the same latent
  wrong positive, behind an export that *is* callable. That is the same shape as #2005 and is
  deliberately left for a separate decision rather than pronounced healthy.
