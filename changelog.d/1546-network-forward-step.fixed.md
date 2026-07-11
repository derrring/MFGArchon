- **FPNetworkSolver.forward_step fails loud instead of silently mis-stepping** (Issue #1546). The
  "interface compatibility" single-step stub never precomputed transition rates, so a fresh solver hit
  the wrong-signed legacy drift, a post-solve call reused stale rates, and it skipped the node-BC /
  #1478 mass-renorm gate (false conservation under an ABSORBING node). It has no callers; it now raises
  NotImplementedError directing callers to solve_fp_system. The legacy fallback in _compute_drift_term
  (reachable only off the solve_fp_system path, which always precomputes) also raises rather than
  resurrecting the #1474 wrong-signed uphill drift.
