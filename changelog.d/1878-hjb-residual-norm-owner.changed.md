- **`hjb_residual_norm` names the grid-scaled HJB residual norm** — `||F||_2 * sqrt(dx)`, previously
  an inline expression in `newton_hjb_step` (Issue #1878). Behaviour is unchanged, pinned on a
  configuration chosen because it is *not* quiet: `Nx=1601`, `FDM_UPWIND`, 2 sweeps, where 95 of
  627592 finite-difference Jacobian probes exceed the `1e6` p-value clip limit, so the clip is
  actively firing while the comparison runs. `U` and `M` are bit-identical across the change
  (`np.array_equal` True, max difference `0.0`), with each run printing whether the module under it
  defines the new function — a before/after in which both sides are the same build is the failure
  mode this control exists to catch, and it caught one.
  The reason it is a function is measured. A second caller written during #1878 restated the norm and
  dropped the `sqrt(dx)`; at `Nx=21` that is a factor of 4.5, and the two paths were being compared
  against each other, so the caller's Armijo test demanded a 4.5x reduction to score as "no worse",
  rejected every step, and returned its input while reporting the residual at the *starting* iterate.
  It presented as a five-order-of-magnitude improvement and as 11 red tests elsewhere.
- **This is a naming, not yet a consolidation, and the docstring says so.** `HJBFDMSolver`'s nD path
  and `HJBGFDMSolver` compare an **unscaled** 2-norm against `DEFAULT_NEWTON_TOLERANCE` — the same
  constant the scaled norm here is tested against — so one tolerance means different things on
  different paths, by a factor of `sqrt(dx)`. Routing them through this function is a behaviour
  change on those paths and is not attempted here. Correcting an earlier version of this entry: it
  claimed "implementations of the quantity 2 -> 1", which does not reproduce — the count is 5.
