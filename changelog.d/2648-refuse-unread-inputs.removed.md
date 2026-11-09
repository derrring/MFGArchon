- **Inputs that never had an effect are removed** (#2648, stage 2; rulings 3 and 7, and the removal before 1.0).
  Breaking (pre-1.0). Passing one raises `TypeError`, and a field or parameter after a removed one is keyword-only,
  so a positional argument past it raises rather than shifting onto the next. What to do instead:
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
  reading it now raises `AttributeError` where it gave `None`, and passing `ergodic_constant=` raises `TypeError`.
