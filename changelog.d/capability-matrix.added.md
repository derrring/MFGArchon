`scripts/capability_matrix.py` — a bidirectional ratchet over what the public solve
surface can actually do. Six configurations run through `problem.solve()` (or a public
constructor) and are checked against an oracle outside the code under test: mass
conservation for `FDM_UPWIND`, `SL_LINEAR`, `FDM_CENTERED` and `FVM_MUSCL`, FVM-vs-FDM
agreement on one 1-D LQ problem, and `HJBGFDMSolver(derivative_method="rbf")`
construction. Baseline in `scripts/capability_baseline.json`; wired into
`scripts/local_ci.sh` (~40 s, not in `--fast`). Refs #1706.
