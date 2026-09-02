- **Removed `TestGPUPerformance::test_gpu_faster_than_cpu_for_large_N`** and its now-empty class.
  Its name claimed a performance result; its only assertion was `speedup > 0.1`, which admits a 10x
  SLOWDOWN, so it could not fail for a correctness defect and could not fail for the performance
  claim either. It was also one of the two tests red since #1921, so it cost a red without buying a
  signal. Nothing outside this file referenced it and no baseline indexes it.

  **The performance claim now has no oracle, and that is the honest close-out rather than a
  hand-off.** `benchmarks/particle_gpu_speedup_analysis.py` looks like the owner and is not: it
  raises `ValueError: u_terminal (terminal condition) must be provided in MFGComponents` on its
  first call, before any timing, and its own `:65` is `TorchBackend(device=device)` at default
  float64 -- the construction this very issue exists to fix. Nothing in any tier runs `benchmarks/`.
  Filed rather than fixed here.

  Also removed with it: the only MPS x `periodic_bc` cell for the particle pipeline. The remaining
  `device="mps"` constructions in `tests/` are the #1921 guard assertion, this file's `no_flux`
  capability cell, and two `TorchKDE` tests that do not exercise the pipeline.

  `test_boundary_conditions_gpu` was examined for the same treatment and KEPT. A mutation ladder
  appeared to show it inert; that was an artifact of injecting symmetrically into a test whose
  assertion is a DIFFERENCE (`periodic` against `no_flux`), which cancels the injection by
  construction. It carries its own positive control on BC-dispatch liveness with a bound
  recalibrated for #2181, and it is the sibling of the #1910 refusal pin.
