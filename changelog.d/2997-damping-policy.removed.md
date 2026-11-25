- **The per-route damping keywords** (#2997). Each is replaced by `damping=`, a `DampingPolicy`; external callers of
  them must migrate.
  - `FixedPointIterator`: `relaxation`, `relaxation_M`, `relaxation_schedule`, `relaxation_schedule_M`,
    `adaptive_relaxation`, `adaptive_relaxation_decay`, `adaptive_relaxation_min`, `use_anderson`, `anderson_depth`,
    `anderson_beta`.
  - `BlockIterator`, `BlockGaussSeidelIterator`, `BlockJacobiIterator`: `relaxation`, `relaxation_M`.
  - `FictitiousPlayIterator`: `learning_rate_schedule`, `initial_learning_rate`, `min_learning_rate`,
    `damp_value_function`, and the method `get_learning_rate`.
  - `NewtonMFGSolver`: `picard_damping`.
  - `GraphMFGSolver`, `RegimeSwitchingIterator`: a number as `damping=` (stage 1b).
  - `MultiPopulationIterator`: `relaxation`, and its attribute `relaxation` (stage 1b).
  - `create_network_mfg_solver`, `create_simple_network_solver`: `relaxation`.
  - `PicardConfig`: `relaxation`, `relaxation_M`, `relaxation_schedule`, `relaxation_schedule_M`, `adaptive_relaxation`,
    `anderson_memory`; replaced by `PicardConfig.damping`.
- **The routes' damping attributes and metadata keys** (#2997), which the policy holds: writing an attribute after
  construction no longer changes the solve.
  - Attributes: `FixedPointIterator`'s `relaxation`, `relaxation_M`, `relaxation_schedule`, `relaxation_schedule_M`,
    `adaptive_relaxation`, `adaptive_relaxation_decay`, `adaptive_relaxation_min`, `use_anderson` and
    `anderson_accelerator`; the Block iterators' `relaxation` and `relaxation_M`; `FictitiousPlayIterator`'s
    `initial_learning_rate`, `min_learning_rate` and `damp_value_function`; `NewtonMFGSolver`'s `picard_damping` and
    `total_iterations`. Each route's `damping` holds its constructor's policy: given none, the Block iterators and
    `NewtonMFGSolver` hold their default, and `FixedPointIterator` and `FictitiousPlayIterator` hold None, since a
    solve's config or the route's default then decides.
  - Result metadata: `FictitiousPlayIterator`'s `learning_rate_schedule` and `damp_value_function`, and the Block
    iterators' `relaxation`, replaced by `damping`, the policy applied; `NewtonMFGSolver`'s `total_iterations`, replaced
    by `SolverResult.iterations`, the sweeps whose change was measured.
- **The initial-density fallbacks** (#2997 ruling 8). The four routes read the problem's `get_m_initial()` and nothing
  else. On `main` the Block iterators and `NewtonMFGSolver` fell back to `problem.m_initial`, then to a uniform density
  with a warning, and `FixedPointIterator` and `FictitiousPlayIterator` read `problem.get_m_init()`; a duck-typed problem
  without `get_m_initial()` now raises `AttributeError` naming it.
- **`warm_start_data["U_prev"]` and `warm_start_data["M_prev"]` on the four routes** (#2997): the arrays are
  `get_warm_start_data()`'s, and `warm_start_data` holds the metadata.
- **The helpers the policy and the iterate's owner replace** (#2997): `apply_damping`, `compute_scheduled_damping`,
  `adapt_damping` (by `DampingRun`); `initialize_cold_start`, `allocate_state_arrays`, `MFGResidual.get_initial_guess`
  (by `IterationState.start`, its allocation `allocate_iterate`); `construct_solver_result`, which nothing called;
  `picard_config_to_iterator_kwargs` (Anderson is the policy's), so `translate_solver_config` returns three dicts; the
  schedule registry `mfgarchon.utils.iteration` (the policy's schedules).
- **`FixedPointIterator.set_warm_start_data(U, M)` and `FictitiousPlayIterator.set_warm_start_data(U, M)`** (#2997),
  with their `get` and `clear` overrides: the base class's `set_warm_start_data((U, M), metadata=None)` replaces them,
  and checks the shapes.
- **The base class's warm-start store** (#2997, stage 1b; the user's ruling of 2026-11-25). On a
  `BaseCouplingIterator` subclass whose `_iteration_state()` returns None, `set_warm_start_data` stored the arrays
  and `get_warm_start_data` returned them; it now raises `NotImplementedError`. `GraphMFGSolver` and
  `RegimeSwitchingIterator` were two such routes, which never read it. A subclass takes a warm start by returning an
  `IterationState` from `_iteration_state` and starting its solve from that state's `start()`.
- **`NewtonMFGSolver.solve()`'s `(U, M, info)` tuple** (#2997 ruling 4): it returns a `SolverResult`, as every route does,
  whose `metadata` is the former `info` and whose error histories are the absolute change per measured sweep.
