- **`RegimeSwitchingIterator` no longer drives a positivity-preserving FP scheme negative** (Issue
  #1681). The FP right-hand side is `sum_{j!=k} Q[j,k] m^j - q_k m^k`; only the first term is
  external to regime `k`. `_make_fp_source` evaluated the second at the previous Picard iterate and
  passed it to the solver as part of `source_term`, where it acted as a sink that is neither
  non-negative nor proportional to the density the current step has. `divergence_upwind` preserves
  positivity against a non-negative source, so it had no guarantee left, and a plain 1-D LQ problem
  reached `-1.0e-03`. Counterfactuals on the fixture: with `Q = 0` the same solve stays at
  `+1.3e-03` with exact mass, with the sink dropped at `+3.4e-03`, and at the failing timestep the
  source is negative at every node. The diagonal is now carried by the substitution
  `m^k = exp(-q_k t) n^k`, which removes it exactly and leaves an inflow that
  `RegimeSwitchingConfig.validate` already guarantees to be non-negative; positivity is structural
  rather than conditional. Measured after: minimum density `+8.4e-04`, stable under `dt`-refinement
  and iteration count, with regime masses tracking the Markov chain's own `M(0) expm(Qt)` to
  `2.2e-03`. Two configurations are now refused rather than silently mis-solved. Inhomogeneous FP boundary
  data is one: the factor is exact only for conditions homogeneous in the density, so with
  `g != 0` the solve would return `g*exp(-q_k t)` at the wall instead of `g` (measured 0.180967
  and 0.163746 against an intended 0.2, at two different rates). Carrying the factor into the
  boundary data is Issue #1805. Both this and the horizon check run at `solve()` as well as at
  construction, because `Q` and the solvers' BCs are mutable and re-read there.
  Because the factor spans `exp(q_k T)`, construction now refuses `q_k * T > 50` rather
  than returning a density that has lost its leading digits. The strict `xfail` in
  `tests/integration/test_phase1_5_validation.py` that pointed at this issue is removed.
