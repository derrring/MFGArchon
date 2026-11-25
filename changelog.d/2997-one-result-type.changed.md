- **One result type for every coupled route** (#2997 rulings 4 and 7, stage 1c). `FixedPointIterator`, the Block
  iterators, `FictitiousPlayIterator`, `NewtonMFGSolver`, `GraphMFGSolver`, `RegimeSwitchingIterator` and
  `MultiPopulationIterator` build their `SolverResult` in one owner (`coupled_result`), which decides the fields every
  route has.
  - **A component axis.** On the last three, U and M are of shape (K, Nt+1, *shape), component k at `U[k]`, and
    `SolverResult.components` is K (None on the four others). `get_results()` on `GraphMFGSolver` and
    `RegimeSwitchingIterator` returns every component's U and M, where it returned component 0's.
  - **The histories.** `error_history_U` and `error_history_M` are each sweep's absolute L2 change, the worst
    component's, on every route; `metadata["l2distu_rel"]` and `metadata["l2distm_rel"]` hold the relative change, now
    on `NewtonMFGSolver` too; `metadata["per_component"]` holds each component's four histories.
  - **A diverged stop.** `iterations` counts every sweep run, the diverged one included, whose history entries are NaN, so
    a diverged solve's `final_error_U` and `final_error_M` are NaN, and its `convergence_reason` names the divergence:
    `"diverged_nan"` on every route, or `FixedPointIterator`'s `"diverged_mass_blowup"`. On `main`, `FixedPointIterator`
    and `FictitiousPlayIterator` published a 1.0 for that sweep, never measured, and so did their
    `get_convergence_data()`; the Block iterators and Newton did not count it; the three component routes counted it
    with no history entry. The arrays are as on `main`: each component the sweep solved holds that HJB output as its U,
    the diverged one's with its NaN kept (#1718), and every other U and every M are the iterate after the last finite
    sweep.
  - **`mass_conservation_error`** is measured on all seven routes: the drift of the mass the route's model conserves.
    That is the total over components on `GraphMFGSolver` and `RegimeSwitchingIterator`, whose components exchange mass,
    and the worst population's on `MultiPopulationIterator`, whose populations exchange none; each component's own drift
    is in `metadata["per_component"]`. RegimeSwitching's Q conserves the total, its rows summing to zero. Graph's
    coupling conserves it where its FP sources sum to zero over the nodes, as the default couplings' do on a symmetric
    adjacency. A directed adjacency or a custom `coupling_fp` changes the total by design, and the drift is then the
    coupling's net source, read against the coupling as a wall's loss is. `FixedPointIterator`'s value is unchanged. On `main` it was None on the Block
    iterators, `FictitiousPlayIterator` and `NewtonMFGSolver`, and absent on the three component routes. An FP solver
    without `mass_conservation_error_override` is measured through the geometry's integral, where `FixedPointIterator`
    raised `AttributeError` once its solve ended.
  - **`convergence_reason`** is the criteria's reason at the stop on the three component routes, which wrote one only
    when a verdict was refused. Their refusal reads "the Picard criteria were met (<the criteria's reason>)", not "()".
    `NewtonMFGSolver`'s stop at a warm-up sweep whose change meets the criteria reads "Picard warm-up: " before the
    criteria's reason, where it read "Converged during Picard warm-up". Its stop at the check after the warm-up, the
    Newton phase's verdict on the residual there, still reads "Converged during Picard warm-up".
  - **`execution_time`** is the solve's time, measured around it on all seven routes. On `main` it was None on the four
    `SolverResult` routes, and the three component routes' results had no such field.
  - **`metadata["damping"]`**, the policy applied, is on all seven routes; on `main`, only fictitious play and the Block
    iterators recorded it. `metadata["component_names"]` names the components: Graph's node indices, the regime
    configuration's `regime_names` (the regime indices where it names none), the populations' names.
  - `SolverResult.save_hdf5` and `load_hdf5` carry `components`.
  - `SolverResult.max_error` is NaN when either final error is, where the builtin `max` dropped a NaN that came second;
    and `compare_to` ranks a NaN final error below any measured one, where every comparison with NaN called a diverged
    solve and a converged one a tie. With `execution_time` measured, `compare_to` names the faster solver when the two
    times differ by more than its 1 ms threshold, where it called every pair of coupled results a tie.
