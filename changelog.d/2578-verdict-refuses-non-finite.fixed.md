- **The outer-tolerance verdict refuses a non-finite change instead of judging around it** (#2578).
  `check_convergence_criteria` took `max` over U's and M's change, and `max(1e-7, nan)` is `1e-7`,
  so a NaN in the density's field was dropped: `(1e-7, nan, 1e-7, nan, 1e-5)` reported converged,
  while the same NaN in U's field reported not converged. It now raises `ValueError` naming every
  non-finite value. Who reaches it:
  - `FixedPointIterator` already stops on a non-finite U or M before judging, and the block,
    fictitious-play, multi-population, graph and regime-switching iterators stop on a non-finite U.
    In those five, a non-finite density from an FP solver that does not check its own output now
    raises instead of reading as converged.
  - `NewtonMFGSolver` judges the map's output after its warm-up and at each Newton iterate (#2565). A
    NaN density at the post-warm-up verdict, at `tolerance=1e3`, reported `Converged during Picard
    warm-up`; it now raises.
  - Finite but overflowing iterates pass those finiteness checks: a density at `1e200` gives an
    `inf`/`nan` change that used to read as converged, and the same in U used to read as not
    converged with an empty reason. Both now raise. `FixedPointIterator` stops a density whose finite
    total mass has grown 1e4-fold before judging (#1489). A density whose total mass overflows to inf
    skips that check, reached its verdict and read as converged, and now raises too.
- **A multi-field solve refuses a non-finite change in any field, naming it** (#2578). The
  multi-population, graph and regime-switching iterators each took builtin `max` over their fields
  before the verdict, and that `max` keeps a NaN only when it comes first, so a NaN density in
  population, node or regime 1 or later reported converged. The three now aggregate through
  `worst_sweep_change` (`mfgarchon.utils.convergence`), which raises `ValueError` naming the value
  and the field, e.g. `l2distm_rel=nan in population k=1`. `refuse_non_finite_change` is the one
  message both refusals use.
