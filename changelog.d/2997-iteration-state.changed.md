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
  - A route built with a damping and solved from a config that sets none keeps its constructor's damping whole. On
    `main`, FixedPoint beside any config took `relaxation` and `relaxation_schedule` from the config, and M's theta
    followed that `relaxation` when `relaxation_M` was unset, so `SolverFactory.create_solver`, which always builds a
    config, dropped the theta it was given. A damping given both to the constructor and in the config
    is refused (the user's ruling of 2026-11-24).
