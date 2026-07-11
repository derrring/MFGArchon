- **Network HJB solvers reject a nonzero `volatility_field` instead of silently ignoring it** (Issue
  #1544). `NetworkHJBSolver` / `NetworkPolicyIterationHJBSolver` have no viscous term — they solve the
  deterministic-control game on the graph and never use the stored graph Laplacian — but accepted a
  `volatility_field` documented "not yet used". A coupled network solve with `D > 0` therefore
  converged to a self-consistent equilibrium of a **mismatched** system (the diffusive network FP
  applies `D*Lap_G(m)` while the HJB solves the `D = 0` game — non-adjoint, consistent only at `D = 0`).
  A nonzero `volatility_field` now raises `NotImplementedError` up front. Stock `NetworkMFGProblem`
  (σ = 0) is unaffected. The natural fix — implementing the `+ D*Lap_G(u)` viscous term to restore
  HJB↔FP duality on graphs — remains open.
