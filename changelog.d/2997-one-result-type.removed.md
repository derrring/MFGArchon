- **`GraphMFGResult`, `RegimeSwitchingResult` and `MultiPopulationResult`** (#2997 ruling 4, stage 1c), with no shim.
  `GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator` return a `SolverResult`, whose U and M carry
  a leading component axis (the `changed` fragment). External callers of them must migrate.
  - `values[k]` is `U[k]`, and `densities[k]` is `M[k]`; MultiPopulation's `U[k]` and `M[k]` read as before.
  - `error_history`, the worst relative change per sweep, is gone: `error_history_U` and `error_history_M` hold the
    absolute change, and `metadata["l2distu_rel"]` and `metadata["l2distm_rel"]` the relative one.
  - MultiPopulation's `errors`, `errors_U` and `errors_M`, the last sweep's relative changes, are the last entries of
    `metadata["per_component"][k]["l2distu_rel"]` and `["l2distm_rel"]`; `errors`, the larger of the two, has no
    replacement.
  - `n_nodes` and `K` are `components`; `population_names` is `metadata["component_names"]`; `regime_config`, the
    caller's own argument, is not carried.
- **`metadata["elapsed_time"]`** on the Block iterators and `NewtonMFGSolver` (stage 1c, the user's ruling): the solve's
  time is `SolverResult.execution_time`, on every coupled route.
- **`NewtonMFGSolver`'s `metadata["converged"]`** (stage 1c, the approver's ruling of 2026-11-25): it copied
  `SolverResult.converged`, which holds the verdict.
- **The Block iterators' `error_history_U` and `error_history_M` attributes**, which nothing read: the result's fields
  of the same names hold the histories.
