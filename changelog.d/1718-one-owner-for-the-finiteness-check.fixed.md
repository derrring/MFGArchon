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

  Pinned by `tests/unit/test_alg/test_diverged_hjb_stops_before_fp_1718.py`. All 10 fail against the
  pre-fix tree, and the 4 call-site tests fail behaviourally, reproducing the CFL misattribution.

  The warning baseline gains one identity, from the new test file: the legacy-API deprecation,
  which is already recorded for many other test files.
