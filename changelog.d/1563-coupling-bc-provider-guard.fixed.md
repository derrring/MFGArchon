- **Coupling loops that cannot resolve dynamic BC providers now fail loud** (Issue #1563, RFC #1574).
  Only `FixedPointIterator` resolves a `BCValueProvider` (e.g. `AdjointConsistentProvider`) stored in a
  `BCSegment.value` — it calls `problem.using_resolved_bc()` each Picard step. The other five coupling
  loops (`FictitiousPlayIterator`, `BlockIterator`/`BlockJacobi`/`BlockGaussSeidel`, `MFGResidual`/
  `NewtonMFGSolver`, `MultiPopulationIterator`, `RegimeSwitchingIterator`) do not, so a provider-based
  boundary condition previously reached the solver unresolved — a deep GFDM row-builder `ValueError`,
  or a silent miss on a non-Robin provider segment. They now raise `NotImplementedError` up front,
  naming the loop, via a single-source guard `assert_bc_providers_resolvable` (one owner in
  `base_mfg.py`, called at construction by all five). `RegimeSwitchingIterator` and
  `MultiPopulationIterator` guard EVERY sub-problem, not just the representative `problems[0]`. This
  adds no coupling behavior and is off published numerics (every published path uses
  `FixedPointIterator`). Wiring the two clean single-problem Picard loops to actually resolve
  providers is a follow-up gated behind a validation experiment.
