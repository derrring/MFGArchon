- **A user abort no longer reports convergence** (Issue #1684, item 2).
  `FixedPointIterator.solve(iteration_callback=...)` let a callback returning `False` set
  `converged = True` with `reason="callback_stopped"`, and it did so *before* the real criteria
  were ever evaluated. Measured on a 21-point no-flux fixture at `tolerance=1e-10`:

  | run | iterations | `converged` | `l2distu_rel` |
  |:--|--:|:--|--:|
  | no callback | 3 | `False` | 3.259e-01 |
  | callback aborts at iteration 1 | 1 | **`True`** | **1.000e+00** |

  The aborted run reported success at three times the error of the run that admitted failure.

  The fix is not "report `False` on abort" — that is one wrong constant replacing another. An abort
  is evidence of neither outcome, so the criteria are now evaluated at that iterate and the flag
  reports what they say; the reason carries the criteria's own verdict when they are met. Aborting
  at a genuinely converged iterate still returns `True`, which is asserted, because a test that
  only checks the failing direction is passed by a hard-coded `False`.

  Mutation-verified, three axes, each killing a different test: restoring the defect reddens 2,
  the hard-coded `False` reddens 1 (the converged-abort direction), and dropping the reason string
  reddens 1.
