- **A diverged HJB is attributed to HJB at every coupling loop** (Issue #1718).

  #1717 fixed one loop; its pre-merge review found the same shape at six more: HJB is solved, an FP
  solve consumes the result, and nothing looks in between. Each failed identically -- the FP solver
  composed a drift from NaN and raised `"Check CFL condition: dt * sigma^2 / dx^2 should be < 0.5"`,
  an FP diagnostic carrying advice that would send a reader to shrink a timestep, for an HJB failure.

  `diverged_value_function` now owns the three decisions that must not vary across sites: publish
  the diverged iterate rather than the last finite one, restore the terminal row, and copy before
  restoring. `fixed_point_iterator`'s own inline guard from #1717 is refactored onto it, so this is
  an owner rather than a seventh copy.

  Wired into `fictitious_play`, `multi_population_iterator`, `graph_mfg_solver`,
  `regime_switching_iterator`, `newton_mfg_solver._run_picard_warmup` and
  `block_iterators._gauss_seidel_step`. Three of those solve every population, node or regime before
  any FP runs, so the check goes inside that loop -- otherwise one population's NaN reaches every
  other one's coupling source before anything notices.

  Two deviations from the design the issue proposed, both forced by measurement: the owner is a
  module function rather than a `BaseCouplingIterator` method, because `multi_population_iterator`
  does not inherit that base; and `value_function_is_finite` is split out because
  `_gauss_seidel_step` needs the question without the answer -- it must skip its FP solve while the
  loop above it publishes and stops.

  Stopping a coupling loop mid-sweep is not just a `break`, and an adversarial review found three
  places where it was treated as one:

  - `NewtonMFGSolver`'s warmup `break` left only the warmup loop; `solve()` then called
    `compute_residual_norm`, which composes an FP solve, so the diverged value function reached FP
    anyway and the site's behaviour was byte-identical before and after. The warmup now reports
    where it diverged and `solve()` returns.
  - `graph_mfg_solver` and `regime_switching_iterator` published `np.empty(0)` / `None` for every
    node or regime not yet solved that sweep, against their own documented `values: list[NDArray]`
    of shape `(Nt+1, Nx)` -- so a consumer looping over `values` to FIND the NaN crashed with
    IndexError before reaching it. Unsolved entries now carry the previous iterate, and densities
    are expanded, since at sweep 0 they are still the 1-D initial conditions.
  - `MultiPopulationIterator` reported the previous completed sweep's near-zero errors on a
    divergence at any sweep after the first, beside an iteration count for the sweep that measured
    nothing -- the #1672 shape. The diverged branch now clears them.

  Pinned by `tests/unit/test_alg/test_diverged_hjb_stops_before_fp_1718.py`, 13 tests. The call-site
  tests fail behaviourally against the pre-fix tree, reproducing the CFL misattribution.

  A second review checked the claim that "each of the four fixes was verified by reverting it alone"
  and found it false for one of them: removing the `errors = []` clearing left the whole CI-marker
  suite green (2741 passed). The fix was right and unpinned, and the sentence asserting it had been
  verified rested on a measurement that did not reproduce. `_LateDivergingHJB` and
  `test_a_late_divergence_does_not_publish_the_previous_sweeps_errors` close that gap -- the earlier
  stub diverged on sweep 0, which is the one case where the pre-loop binding already covered it, so
  the defect lived entirely on the other side of that branch.

  The terminal-condition restore -- one of the owner's three invariants -- was previously unpinned:
  the fixture's terminal was zero and the stub returned zeros, so the assertion could not separate
  "restored" from "never written", and replacing `U_terminal` with `None` at every wired site left
  the suite green. The fixture's terminal is now 4.0 against a stub returning 7.0; that same
  mutation now reddens three tests.

  The warning baseline gains one identity, from the new test file: the legacy-API deprecation,
  which is already recorded for many other test files.
