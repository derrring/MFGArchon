`scripts/capability_matrix.py` — a bidirectional ratchet over what the solve surface
can actually do. Seven configurations are driven end to end and checked against an
oracle outside the code under test: mass conservation for `FDM_UPWIND`, `SL_LINEAR`,
`FDM_CENTERED` and `FVM_MUSCL`, FVM-vs-FDM agreement on one 1-D LQ problem,
two-regime `RegimeSwitchingIterator` non-negativity, and
`HJBGFDMSolver(derivative_method="rbf")` construction. Baseline in
`scripts/capability_baseline.json`; wired into `scripts/local_ci.sh` (~41 s, not in
`--fast`). Refs #1706.
