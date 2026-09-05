- **A network problem's mass is the node sum, not `1/N`** (Issue #2177). `_measure_initial_density`
  gated its network branch on `self.dimension`, but `NetworkMFGProblem` sets `dimension = "network"`
  *after* `super().__init__()` and the measurement runs inside it — so the guard read `dimension == 2`,
  a network problem describing itself as two-dimensional during its own construction. It fell through
  to `point-average`, `sum(m) / num_spatial_points`.

  Three symptoms, one cause. On a 5x5 `GridNetwork` whose density already summed to 1,
  `problem.initial_mass` reported **0.04** under the name "initial density mass"; the #1887 warning
  fired as a false positive with a remedy that could not work, since dividing a density that already
  sums to 1 by its integral changes nothing; and the `node-sum` branch written for exactly this case
  never executed. `NetworkMFGProblem` is the only network problem class, so that branch was dead as
  written. Now: `initial_mass = 1.0`, `initial_mass_measure = 'node-sum'`, no warning.

  The gate reads `self.is_network` — which derives from `geometry.geometry_type` and is already
  correct at measure time — rather than a new predicate, because that question already had an owner.
  Same lesson as #2157: gate on the thing you are about to use.

  **The constructor ordering is untouched.** `topology.py` carries a comment explaining why
  `dimension` is set after `super().__init__()` (`_detect_solver_compatibility()` has already run),
  and moving it was the other candidate fix; this one does not require it.
