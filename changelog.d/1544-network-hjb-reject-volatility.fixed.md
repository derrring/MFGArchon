- **Network HJB solvers reject a nonzero `volatility_field` instead of silently ignoring it** (Issue
  #1544). `NetworkHJBSolver` / `NetworkPolicyIterationHJBSolver` have no viscous term — they solve the
  deterministic-control game on the graph and never use the stored graph Laplacian — but accepted a
  `volatility_field` documented "not yet used". A coupled network solve with `D > 0` therefore
  converged to a self-consistent equilibrium of a **mismatched** system (the diffusive network FP
  applies `D*Lap_G(m)` while the HJB solves the `D = 0` game — non-adjoint, consistent only at `D = 0`).
  A nonzero `volatility_field` now raises `NotImplementedError` up front. Stock `NetworkMFGProblem`
  (σ = 0) is unaffected. **Scope**: this catches only a volatility explicitly threaded into
  `solve_hjb_system`; the graph coupler (`graph_mfg_solver`) does not thread one, so the *stock*
  coupled mismatch — `FPNetworkSolver`'s default `D = 0.1` against the `D = 0` HJB — is still reachable
  and is NOT rejected here (guarding it would break every network coupled solve). Fully closing #1544
  needs the natural fix: implement the `+ D*Lap_G(u)` viscous term to restore HJB↔FP duality on graphs
  (open, blocked on #1470-C).
