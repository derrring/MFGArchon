- **Capability matrix** (`scripts/capability_matrix.py`, Refs #1706) — a bidirectional
  ratchet over what the solve surface can actually do, checked against oracles outside
  the code under test. Seven cells: mass conservation for `FDM_UPWIND`, `SL_LINEAR`,
  `FDM_CENTERED` and `FVM_MUSCL`; FVM-vs-FDM agreement on one 1-D LQ problem;
  two-regime `RegimeSwitchingIterator` non-negativity; and
  `HJBGFDMSolver(derivative_method="rbf")` construction. Baseline in
  `scripts/capability_baseline.json`. Wired into `scripts/local_ci.sh` (~41 s, not in
  `--fast`) and into `nightly.yml` as a standalone job that also runs `--self-test`.
