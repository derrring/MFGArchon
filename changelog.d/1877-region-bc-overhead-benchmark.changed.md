- **A wall-clock performance assertion no longer gates the suite** (Issue #1877).
  `test_region_based_bc.py::TestRegionBasedBCPerformance::test_region_lookup_overhead` failed the
  nightly full run at *"Region-based BC overhead 12.3% exceeds 10%"*. It was measuring the runner.
  The estimator timed 100 standard BC applications, then 100 region-based ones, and divided — two
  blocks separated in time, so anything the machine did in between was charged to the region path.
  Measured over 21 repeats: min **-59.5%**, median 1.0%, max **+105.3%**, sd 34.7pp, exceeding its
  own 10% bound **24%** of the time. An overhead of -59.5% is not a number the region path can
  produce, which is what identifies the estimator rather than the code as the subject.
  The counterfactual — same machine, same load (load average 42), same work, only the estimator
  interleaved so drift cancels — gives min -4.3%, median **0.4%**, max 4.2%, sd 2.0pp. So the
  variance came from the estimator's structure, and loosening the threshold would have hidden that
  rather than fixed it. (The assertion had drifted from its own docstring in the other direction
  already: the docstring promised `<5%` and the assertion allowed `10%`, from the first commit.)
  Interleaving is still not enough to gate on. In a fresh short-lived process the **first** repeat is
  systematically the worst — 14.5%, 9.5%, 11.0% in three of six runs — and single repeats excurse to
  `-35%` and `+713%` even after warmup, so medians above 5% do occur under contention. The true
  overhead is **0.1-1%**; no threshold both admits that and survives the noise. The measurement moved
  to `benchmarks/benchmark_region_bc_overhead.py`, warmed up properly and reporting a median over 9
  repeats, and the test is gone.
  **The local gate never ran it.** It carries the `slow` marker, so `-m "not slow"` deselects it and
  `./scripts/local_ci.sh` skips it; verified by listing the selected set, and by the counts moving
  `6384 -> 6383` collected while the *selected* count stayed at `6226` and deselected went
  `158 -> 157`. Nightly is the only tier that runs `@slow`, so it is the only tier that could have
  failed — the local gate was never lucky, it was structurally blind. (Where the `slow` marker comes
  from is unresolved: `tests/conftest.py:100` adds it only on the name substrings `large`/`slow`/
  `benchmark`, none of which `test_region_lookup_overhead` contains, the file carries no
  `@pytest.mark.slow`, and neutering that conftest branch leaves the test deselected. Noted rather
  than guessed at.)
  **What the suite loses**: nothing it was reliably providing — a 24% false-positive rate against a
  real signal of 1% is not a regression detector. **What is now unguarded**: a change that makes the
  region path materially slower. Nothing catches that automatically any more, and the benchmark has
  to be run deliberately — stated here rather than left for someone to discover.
