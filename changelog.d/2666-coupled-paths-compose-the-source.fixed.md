- **Every coupled solve path applies the problem's own source** (Issue #2666). `MultiPopulationIterator`,
  `RegimeSwitchingIterator` and `CommonNoiseMFGSolver` dropped a declared `source_term_hjb`, `source_term_fp`
  or `nonlocal_operator` on both equations, without a warning, and solved a problem without them. They now
  compose it through the same owner as the single-population loops:
  - a multi-population run composes each population's source;
  - a regime adds its source to the cross-regime term, with the FP side under the regime's integrating factor;
  - each noise path's conditional problem is the parent rebuilt through `MFGProblem._rebuilt`, which now keeps a
    record of the `components=` form's inputs too.
- **Two refusals where there were silent drops.**
  - `BaseHJBSolver.solve()`, the standalone HJB entry, passes no source, so it now raises on a problem that
    declares one, naming the coupled solvers and `solve_hjb_system(source_term=)` as the routes.
  - A `StochasticMFGProblem` that declares `state_penalty` now raises at `create_conditional_problem`, where the
    penalty was dropped: each conditional problem solves with `conditional_hamiltonian`, which has no potential to
    compose it into (#2700). The message names the route that works, folding the penalty into
    `conditional_hamiltonian`.
- **A solver that cannot take a source now says so** on these paths. Of the exported FP solvers, `FPNetworkSolver`
  and `FPGFDMSolver` do not name `source_term`. A multi-population run with one of them now raises
  `NotImplementedError` when the population declares a source; it used to pass none. A regime run, which always
  hands its FP solver the inflow, raises the same error where it used to fail with a `TypeError`.
