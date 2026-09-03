- **The two MPS tests now say which half of the #1921 trade-off they want** (Issue #2210). Both
  constructed `TorchBackend(device="mps")` at the default `precision="float64"`, which the #1921
  guard refuses -- MPS has no float64 at all, and narrowing in silence would put this library's
  1e-10 tolerances out of reach in principle. The guard was doing its job; the two call sites were
  never updated. They now pass `precision="float32"` explicitly. Verified: the tensors land on
  `mps:0` at `torch.float32` rather than falling back to CPU, and the guard still raises for
  float64-on-MPS.

  Neither was reached by any automatic tier: `pytestmark = pytest.mark.optional_torch`, and the
  gate (`scripts/ci_markers.txt`) and `nightly.yml:135` both deselect that marker, so the family
  runs only on a release -- and the last release was 2026-07-04. Measured 2026-09-02: the family
  was `2 failed, 21 passed` and is now green. Whether the marker gets a tier is item 3 of #2210
  and stays open.

  **Close-out (AGENTS.md § *Closing out a fix*): category 3, a capability cell, for
  `test_gpu_pipeline_runs_without_errors` -- it answers "does this configuration run at all".
  It does not certify any number the configuration produces.** `test_gpu_matches_cpu_numerically`
  is renamed `test_torch_cpu_matches_numpy` because neither side was ever a GPU (the comparand is
  `TorchBackend(device="cpu")`), and it closes out as **category 4, an admitted happy-path
  assertion**: its discrimination is not established and its tolerances are not defended. A first
  attempt to characterise both by mutation was withdrawn before merge -- the injection point was
  not the path the prose named, so every figure it produced was a fact about the injector. What
  those tests actually discriminate is filed as #2223, not claimed here.
