- **The per-route damping keywords** (#2997). Each is replaced by `damping=`, a `DampingPolicy`; external callers of
  them must migrate.
  - `FixedPointIterator`: `relaxation`, `relaxation_M`, `relaxation_schedule`, `relaxation_schedule_M`,
    `adaptive_relaxation`, `adaptive_relaxation_decay`, `adaptive_relaxation_min`, `use_anderson`, `anderson_depth`,
    `anderson_beta`.
  - `BlockIterator`, `BlockGaussSeidelIterator`, `BlockJacobiIterator`: `relaxation`, `relaxation_M`.
  - `FictitiousPlayIterator`: `learning_rate_schedule`, `initial_learning_rate`, `min_learning_rate`,
    `damp_value_function`, and the method `get_learning_rate`.
  - `NewtonMFGSolver`: `picard_damping`.
  - `create_network_mfg_solver`, `create_simple_network_solver`: `relaxation`.
  - `PicardConfig`: `relaxation`, `relaxation_M`, `relaxation_schedule`, `relaxation_schedule_M`, `adaptive_relaxation`,
    `anderson_memory`; replaced by `PicardConfig.damping`.
- **The helpers the policy and the iterate's owner replace** (#2997): `apply_damping`, `compute_scheduled_damping`,
  `adapt_damping` (by `DampingRun`); `initialize_cold_start`, `allocate_state_arrays`, `MFGResidual.get_initial_guess`
  (by `IterationState.start`, its allocation `allocate_iterate`); `construct_solver_result`, which nothing called;
  `picard_config_to_iterator_kwargs` (Anderson is the policy's), so `translate_solver_config` returns three dicts; the
  schedule registry `mfgarchon.utils.iteration` (the policy's schedules).
- **`FixedPointIterator.set_warm_start_data(U, M)` and `FictitiousPlayIterator.set_warm_start_data(U, M)`** (#2997),
  with their `get` and `clear` overrides: the base class's `set_warm_start_data((U, M), metadata=None)` replaces them,
  and checks the shapes.
- **`NewtonMFGSolver.solve()`'s `(U, M, info)` tuple** (#2997 ruling 4): it returns a `SolverResult`, as every route does,
  whose `metadata` is the former `info` and whose error histories are the absolute change per measured sweep.
