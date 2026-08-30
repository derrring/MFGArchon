- **Performance monitoring reports the solve's actual verdict** (Issue #1684, item 4).

  Both convergence read sites in `utils/performance/monitoring.py` read `convergence_achieved`
  off the solver object. No class in the package has ever assigned that attribute -- four read
  sites, zero writes, while its sibling `iterations_run` is real and set in `block_iterators.py`
  and `fictitious_play.py`. It was never a rename; it is the unimplemented half of a pair.

  The decorator read it through a defaulting `getattr`, so every tracked run was recorded as
  `converged: False` beside a genuine iteration count, whatever the solve did. `benchmark_solver`
  read it through `hasattr`, which was always false, so no verdict was recorded at all -- and that
  loop discarded `solver.solve()`'s return value, which is where the verdict lives.

  Both now read `converged` from the result, through one private helper so that "what is this
  result's verdict" has a single owner rather than a copy per call site. Where the result carries
  no verdict the key is omitted rather than defaulted: absent means not measured, `False` means
  measured and negative, and the old code could only ever produce the second. The decorator does
  not raise on an unfamiliar return value either -- it wraps arbitrary callables and is not
  entitled to crash what it observes.

  Pinned by `tests/unit/test_utils/test_monitoring_reads_the_result_verdict_1684.py`, of which 5
  of 6 fail against the pre-fix source. The survivor is the non-converged case, which the old code
  got right by accident.

  The third read site, `stochastic/common_noise_solver.py:438`, raised `AttributeError` rather
  than reporting a wrong verdict and is corrected here too. It has never been reached: that solver
  fails earlier, in `create_conditional_problem` (#2191).
