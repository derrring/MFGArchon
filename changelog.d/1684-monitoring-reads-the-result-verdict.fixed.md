- **Performance monitoring reports the solve's actual verdict** (Issue #1684, item 4).

  Both convergence read sites in `utils/performance/monitoring.py` read `convergence_achieved`
  off the **solver** object. No solver class has ever carried it, so both were reading the wrong
  object; and nothing in the package carries it at all today.

  It was a real field once. `SolverResult.convergence_achieved` was a declared, assigned dataclass
  field until `da7b8dfc` (2025-10-08) renamed it to `converged`, with a deprecated property that
  `53a79ddd` (2025-12-06) removed. Three logical read sites remain in the package (four grep
  lines -- #1684 item 4 counts three, and the two numbers are of different things); `common_noise_solver.py` is a call site that rename left behind, written one day before
  it. Three further reads survive in `examples/` notebooks and are not fixed here.

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

  The third read site, `stochastic/common_noise_solver.py:438`, would raise `AttributeError`
  rather than report a wrong verdict, and is corrected here too. It has never been reached: that solver
  fails earlier, in `create_conditional_problem` (#2191).
