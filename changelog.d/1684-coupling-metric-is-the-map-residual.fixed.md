- **The Picard convergence metric measures the map's residual, not the damped step** (Issue #1684,
  item 7 — the second of the two the issue separates out as able to change published numbers).
  `FixedPointIterator` computed `calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, ...)`
  from the iterate *after* damping or Anderson, so the number the convergence criteria read carried
  the damping factor. `l2distu_abs` at iteration 2 on a 21-point no-flux fixture, before:

  | relaxation | 1.0 | 0.5 | 0.2 | 0.1 |
  |:--|--:|--:|--:|--:|
  | reported | 5.825e-01 | 3.357e-01 | 1.516e-01 | 7.944e-02 |

  A factor of 7.3 bought by turning damping down. Convergence is a property of the map — the fixed
  point is where `Φ(U) = U` — while damping is a property of the path, and measuring the damped step
  conflates *we stopped moving* with *we arrived*. After the fix the same column runs the other way
  (5.825e-01 → 7.944e-01): heavier damping has made less progress and now says so.

  **Blast radius zero** — 6029 passed, gate green — the same finding as item 6 one level up: the
  metric every coupled solve reports has been damping-scaled for the lifetime of the iterator with
  nothing able to tell.

  Pinned on the exact law rather than an ordering: at iteration 1 every damping starts from the same
  initial guesses, so the map output is identical and both metrics must be too — measured
  8.356340634e-01 and 6.806196935e-01 at four dampings, to every digit. Mutation-verified, and the
  second mutation is why the law is stated for both fields: a half-fix leaving `M` damped passes any
  U-only assertion.
