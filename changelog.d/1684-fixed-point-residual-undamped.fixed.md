- **`FixedPointSolver`'s residual no longer shrinks with the damping factor** (Issue #1684, item 6).
  The residual of `x = G(x)` is `||G(x) - x||`. The solver measured `||x_updated - x_current||`
  where `x_updated` is the value *after* under-relaxation, which equals `relaxation * ||G(x) - x||`
  — so turning damping down made anything converge.

  Measured on `G(x) = 0.9x + 1` (fixed point 10) from `x0 = 0` at `tolerance = 1e-2`, before:

  | relaxation | `converged` | iters | reported | x returned | true `\|G(x)-x\|` |
  |--:|:--|--:|--:|--:|--:|
  | 0.1 | `False` | 200 | 1.353e-02 | 8.660203 | 0.133980 |
  | 0.01 | **`True`** | **2** | 9.990e-03 | **0.019990** | 0.998001 |
  | 0.001 | **`True`** | **1** | 1.000e-03 | **0.001000** | 0.999900 |

  Success at `x = 0.02` and failure at `x = 8.66`, on the same problem; the 0.001 run declared
  convergence after one iteration, barely off its initial guess. After the fix the reported value
  equals the true `|G(x) - x|` in every row, and heavy damping honestly reports that it has not
  converged within the budget.

  **Blast radius zero** — 6022 passed, gate green — which is itself the finding: the residual has
  been scaled by the damping factor for as long as this solver has existed and no test noticed.
  Pinned now by `tests/unit/test_utils/test_fixed_point_residual_is_undamped_1684.py`, whose law is
  that the first residual is `|G(x0) - x0|` at every damping; mutation-verified, both restoring the
  defect and measuring the step a different wrong way redden 6 of its 7 cases.
