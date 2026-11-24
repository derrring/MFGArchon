- **One owner of the coupled iterate** (#2997, stage 1a). `IterationState` sizes the iterate, cold- or warm-starts it,
  writes level 0 and the terminal row back after each sweep, and packs it for Newton, for `FixedPointIterator`, the
  Block iterators, `FictitiousPlayIterator` and `NewtonMFGSolver`. Their output is unchanged except as follows.
  - `NewtonMFGSolver` starts from the shared cold start (u(T) at m(0) in every row of U; m(0), with the FP's prescribed
    values, at every level of M) instead of zeros, so its warm-up path and its iteration counts change.
  - Newton reads u(T) with the HJB solver's terminal binding, so a terminal coupling declared off a grid solves on
    Newton instead of being refused.
  - The Block iterators' `error_history_U` and `error_history_M` are the absolute L2 change per sweep, as on every
    route; the relative change is in their metadata.
  - FixedPoint's adaptive damping with M's theta unset adapts each field on its own history; M's theta no longer
    follows U's adaptation.
  - Fictitious play's `"constant"` schedule is alpha0 itself, not 0.5 alpha0.
