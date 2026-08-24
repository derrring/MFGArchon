- **BREAKING.** `AdvancedBoundaryHandler.apply_robin_bc` removed from the boundary protocol.
  Removing a member from a publicly exported `@runtime_checkable` protocol changes `isinstance`
  semantics for downstream objects and breaks static narrowing for any external consumer, so it is
  marked breaking even though nothing in this repository implements or calls it — verified across
  the full git history. It goes for three reasons: it was never implemented anywhere; it duplicates
  `BoundaryHandler.apply_boundary_conditions(values, bc, time)`, which already carries Robin
  through `BCSegment.alpha`/`.beta`; and its scalar `alpha: float, beta: float` is the shape #1957
  rejected — the Robin coefficient of a reflecting FP wall is `D_pH(x, grad u) . n`, which varies
  along the boundary and is recomputed every Picard iterate, so a float cannot hold it. Note the
  reason is *not* that a `values -> values` signature is impossible: `_apply_robin_along_normal`
  (`applicator_implicit.py`) and the `BCType.ROBIN` branch of `applicator_meshfree.py` both enforce
  Robin exactly that way.
