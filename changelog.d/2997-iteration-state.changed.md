- **One owner of the coupled iterate** (#2997, stages 1a and 1b). `IterationState` sizes the iterate, cold- or
  warm-starts it, writes level 0 and the terminal row back after each sweep, and packs it for Newton, for
  `FixedPointIterator`, the Block iterators, `FictitiousPlayIterator` and `NewtonMFGSolver`, and once per component
  for `GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator`. Their output is unchanged except as
  follows.
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
  - `GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator` (stage 1b) start U from u(T) at m(0) in
    every row, and damp U and M from the first sweep under their default, `DampingPolicy.constant(0.5)`, so their
    finite-sweep results and iteration counts change. `MultiPopulationIterator` damps U, which it took whole;
    `RegimeSwitchingIterator` damps M at its first sweep, which it took whole, and judges that sweep, so its
    `error_history[0]` is the first sweep's change instead of `inf`, and a tolerance the first sweep meets stops it.
    Each one's old damping is one explicit `DampingPolicy` away; its U cold start and the unjudged first sweep are not.
  - `GraphMFGSolver` and `RegimeSwitchingIterator` write U's terminal row back to the u(T) each sweep's HJB solve was
    given, which a terminal coupling moves with m(T), instead of the damped mix of two sweeps' u(T); and they read u(T)
    with each component's HJB solver's terminal binding, so a terminal coupling declared off a grid solves on them
    instead of being refused.
  - The three solve on a 2-D grid, where each raised `ValueError` writing a level of the grid's shape into a row sized
    from its first axis
    (`MultiPopulationIterator` with one population; with two or more it refuses a grid of two or more dimensions by
    name, #2335). The graph couplings' and the regime coupling's sources read each level at the geometry's points, in
    its order (`IterationState.level_at_points`); a custom `coupling_hjb` or `coupling_fp` is handed each level as the
    iterate holds it, of the grid's shape.
  - `GraphMFGSolver.set_warm_start_data` and `RegimeSwitchingIterator.set_warm_start_data` raise
    `NotImplementedError`: each accepted a warm start and never read it (the user's ruling of 2026-11-25).
  - `NewtonMFGSolver` judges each warm-up sweep's change, FixedPoint's sweep, through the one measure, as the Newton
    phase judges the residual it solves. The first warm-up sweep whose change meets the criteria ends the solve there
    with that sweep's iterate: converged, unless its HJB steps failed (#1878). `main` ran every warm-up sweep, then
    judged the residual at their output and went on into Newton steps when that failed. So the count changes whenever
    a warm-up sweep's change meets the criteria, and the iterate too, unless that sweep is the warm-up's last and main's
    check at its output passed.
  - `GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator` refuse a `PointCloudGeometry`, on which
    each solved at `main`, as the other four routes do: the iterate's owner takes a tensor grid, or an unstructured mesh
    where the route allows one (the user's ruling of 2026-11-25).
  - `BaseCouplingIterator.solve`'s abstract signature is `solve(self)`: each route's solve adds its own parameters, all
    optional.
