- **Inputs that never had an effect are refused when passed, by name, until v0.25.0** (#2648, stage 2; rulings 3 and
  7). Breaking (pre-1.0). Each refusal names the class or function that refuses it, says the input never had an
  effect, and says what the code does instead. Each input keeps its place in the signature, and its default is
  `NEVER_READ`. So the classes still construct, `dataclasses.replace` still works, and reading one of these
  attributes gives `NEVER_READ`. `NEVER_READ` is not JSON-serializable, so `json.dumps(dataclasses.asdict(cfg))` on a
  dataclass with a refused field raises `TypeError` (on `MCMCConfig()` and `MCConfig()` it worked before): drop the
  refused fields before serializing.
  - `MCMCConfig`: `num_chains`, `max_tree_depth`, `max_adapt_iter`, `parallel`, `compute_rhat`,
    `effective_sample_size`. Every sampler runs one chain; `compute_rhat(chains)` is a function to call on samples
    from several runs, and `effective_sample_size(chains)` one to call on the samples.
  - `MCConfig`: `adaptive`, `max_adaptive_samples`, `proposal_distribution`, `batch_size`, `chunk_size`, `parallel`.
    `adaptive_monte_carlo` takes its own `batch_size=` and `max_samples=`.
  - `WassersteinSolverConfig`: `step_size`, `relative_tolerance`, `history_length`.
  - `NetworkMFGComponents`: `velocity_space_dim`, `trajectory_cost_func`, `relaxed_control`, `edge_cost_func`,
    `congestion_func`, `edge_interaction_func` and `problem_params`. No network solve reads problem parameters:
    close over the values in your callables.
  - `VariationalMFGComponents`: `terminal_cost_dx_func`, `potential_func`, `potential_dx_func`, `problem_type`. A
    potential belongs in `lagrangian_func`.
  - `MultiPopulationProblem(cross_coupling=)`, `GraphBCConfig(default_bc_type=)`, `HybridMazeConfig(radial_center=)`,
    `SweepConfiguration(batch_size=)`.
  - `HJBGFDMSolver(qp_usage_target=)`, `PerformanceHook(track_detailed_timing=)`,
    `MFGNotebookReporter(enable_latex=, template_style=)`, and `potential_func=` in `LQPolicyIterationHelper` and
    `create_lq_policy_problem`.
- **`SolverResult.ergodic_constant` is removed** (#2652). No solver computed one, and there is no stationary mode, so
  reading it now raises `AttributeError` where it gave `None`, and passing `ergodic_constant=` is refused by name.
- **A call that does not bind** a constructor refusing an input like these now gets the constructor's own
  `TypeError`, with no binding error chained to it.
