- **A wall-clock performance assertion no longer gates the suite** (Issue #1877).
  `test_region_based_bc.py::TestRegionBasedBCPerformance::test_region_lookup_overhead` failed the
  nightly full run at *"Region-based BC overhead 12.3% exceeds 10%"*. It was measuring the runner.
  The estimator timed 100 standard BC applications, then 100 region-based ones, and divided — two
  blocks separated in time, so anything the machine did in between was charged to the region path.
  Over 21 repeats: min **-59.5%**, median 1.0%, max **+105.3%**, sd 34.7pp, exceeding its own 10%
  bound **24%** of the time. An overhead of -59.5% is not a cost the region path can have, which is
  what identifies the estimator rather than the code as the subject. (The assertion had never matched
  its own docstring either: `"""...<5% overhead"""` over `assert overhead < 10`, from the first
  commit.)
- **No form of this measurement reaches the precision the assertion needed.** Interleaving the two
  variants fixes the drift defect — within one process, 11 repeats give min -4.3%, median 0.4%, max
  4.2%, sd 2.0pp. That figure does not transfer: repeats inside one process share warm caches, one
  allocation layout and one core assignment, and the ratio itself shifts between processes. Across
  **14 fresh invocations at load average 2.0**, an idle machine, the median-of-9 headline ran from
  **-13.33% to +13.25%**, sd 5.50, with 86% of invocations outside the 0.1-1% band that repeated
  measurement puts the true cost in. Taking the ratio of per-variant minima instead, on the theory
  that noise is one-sided, does not help either: 16 fresh processes, sd 5.60 against the median's
  5.58. So the 10% threshold was unreachable by any statistic, and loosening it would have been
  guesswork rather than a fix.
  The measurement moved to `benchmarks/benchmark_region_bc_overhead.py`, which prints the load
  average, the per-repeat values and the between-process spread, and says in as many words that one
  invocation is not an answer. The test is gone.
- **The local gate never ran it.** The test carried `@pytest.mark.slow` (`test_region_based_bc.py:278`,
  removed by this diff along with the test), so `-m "not slow"` deselected it and
  `./scripts/local_ci.sh` skipped it; nightly is the only tier that runs `@slow`. Confirmed by the
  counts: collected `6384 -> 6383` while the *selected* count stayed at `6226` and deselected went
  `158 -> 157`. The local gate was not lucky — it was structurally blind, which is the whole reason
  the failure could only appear at night.
- **What the suite loses**: nothing it was reliably providing — a 24% false-positive rate against a
  real signal near 1% is not a regression detector. **What is now unguarded**: a change that makes
  the region path materially slower. Nothing catches that automatically any more, and the benchmark
  must be run deliberately and repeatedly. Stated here rather than left to be discovered.
