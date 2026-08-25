- **BREAKING.** `mfgarchon.geometry.boundary.protocols.AdvancedBoundaryHandler` removed. Completes the
  removal begun by `remove-apply-robin-bc-protocol-member`: with `apply_robin_bc` gone the protocol
  added exactly one member over `BoundaryHandler` — `get_boundary_normals` — and **nothing
  implemented it.** Measured by sweeping the package: **371 modules, 631 classes** (classes defined
  under `mfgarchon`, deduped; 0 import failures), zero satisfying it, zero `isinstance`/`issubclass`
  sites and zero call sites *in this repository* — out-of-repo sites are why this is marked breaking.
  **Three import paths break, not one**: `from mfgarchon.geometry.boundary import
  AdvancedBoundaryHandler` (the lazy export), `from mfgarchon.geometry.boundary.protocols import
  AdvancedBoundaryHandler` (the class was defined at module level), and `import *` from
  `...boundary.protocols`, which that module's `__all__` advertised. The symbol shipped in **v0.21.0**,
  the latest published release. The one member it added has no agreed signature either — three classes
  *define* a `get_boundary_normals` and no two agree: `CollocationPointSet` `(self)`,
  `MeshfreeApplicator` `(self, points)`, `Hyperrectangle` `(self, points, eps)` (a fourth,
  `ImplicitApplicator`, inherits `MeshfreeApplicator`'s, so a `hasattr` sweep returns four where this
  count is three). The protocol declared the first shape and matched none of the classes. Being
  `@runtime_checkable`, which compares names and not signatures, it was also a latent wrong positive:
  `Hyperrectangle` already matched three of its four members and was one attribute away from silently
  classifying as a solver boundary handler.
- **Deliberate deviation from the deprecation policy's clause (4), stated rather than passed over.**
  The symbol was released and is removed here with no deprecation window. Clause (1) — "old API calls
  new internally, zero behavior difference" — cannot be satisfied: the only candidate alias is the
  protocol this one extends, and aliasing to it would *widen* `isinstance` to accept objects lacking
  `get_boundary_normals`, which is a behaviour difference by construction. With no redirect
  constructible, clause (4)'s timeline would mean keeping a protocol nothing implements for three
  minor versions. This follows `remove-apply-robin-bc-protocol-member` (#2094, one commit earlier),
  which removed a member of this same protocol on the same terms. Noted also because #2078's stated
  delete-criteria include "not surfaced in the package namespace", and
  `hasattr(mfgarchon.geometry.boundary, "AdvancedBoundaryHandler")` **was** True before this change —
  that criterion is not met here, and the BREAKING marker is what stands in for it.
- **Unchanged, and not to be confused with it:** `geometry.boundary.protocols.BoundaryHandler`. Three
  distinct live classes carry the bare name `BoundaryHandler` — this protocol, the concrete
  `alg.numerical.gfdm_components.BoundaryHandler` (public in that package's `__all__` and instantiated
  at `hjb_gfdm.py:900`), and the ABC in `gfdm_strategies.py`. **Only the protocol is discussed here,
  and none of the three is removed.** The protocol keeps one implementer, `HJBSemiLagrangianSolver`;
  its only consumer `validate_boundary_handler` is called nowhere outside an `if __name__ ==
  "__main__"` block; and `Hyperrectangle` matches two of its three members with incompatible
  signatures — the same latent wrong positive, behind an export that *is* callable. That is the shape
  of #2005 and is left for a separate decision rather than pronounced healthy.
