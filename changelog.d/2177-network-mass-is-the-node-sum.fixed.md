- **A network problem's mass is the node sum, not `1/N`** (Issue #2177). `_measure_initial_density`
  gated its network branch on `self.dimension`, but `NetworkMFGProblem` sets `dimension = "network"`
  *after* `super().__init__()` and the measurement runs inside it — so the guard read `dimension == 2`,
  a network problem describing itself as two-dimensional during its own construction. It fell through
  to `point-average`, `sum(m) / num_spatial_points`.

  Three symptoms, one cause. On a 5x5 `GridNetwork` whose density already summed to 1,
  `problem.initial_mass` reported **0.04** under the name "initial density mass"; the #1887 warning
  fired as a false positive with a remedy that could not work, since dividing a density that already
  sums to 1 by its integral changes nothing; and the `node-sum` branch never executed on this path.

  **The branch was not dead in general.** `_init_network` sets `dimension = "network"` before
  `_initialize_functions` runs, so `MFGProblem(network=<graph>)` always reached it. What was
  unreachable is the *geometry-first* route, and both of its entry points move:

  | | before | after |
  |:---|:---|:---|
  | `NetworkMFGProblem(GridNetwork)` | `point-average` 0.04 + warning | `node-sum` 1.0, no warning |
  | `MFGProblem(geometry=<NetworkGeometry>)` | `point-average` 0.04 + warning | `node-sum` 1.0, no warning |
  | `MFGProblem(network=<nx.Graph>)` | `node-sum` — already correct | unchanged |

  #2177's body states the branch was dead; that is true only of the path it examined, and the second
  row above is a fix the issue does not mention.

  **`node-sum` is the measure the solve itself uses.** `alg/numerical/network_solvers/fp_network.py`
  takes `float(np.sum(M[0, :]))` as its own total mass, with a comment stating that the node masses
  *are* the mass functional and "the ratio needs no weights". So this puts `problem.initial_mass` on
  the same functional the network FP solver conserves, rather than merely on a different one that is
  less wrong.

  The gate reads `self.is_network` — which derives from `geometry.geometry_type` and is already
  correct at measure time — rather than a new predicate, because that question already had an owner.
  Same lesson as #2157: gate on the thing you are about to use.

  **The constructor ordering is untouched.** `topology.py` carries a comment explaining why
  `dimension` is set after `super().__init__()` (`_detect_solver_compatibility()` has already run),
  and moving it was the other candidate fix; this one does not require it.
