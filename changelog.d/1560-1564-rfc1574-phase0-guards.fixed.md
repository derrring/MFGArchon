- **RFC #1574 Phase-0 capability-honesty guards** (Issues #1560, #1564). Two solvers declared/dispatched
  a boundary-condition surface broader than the code that honors it and silently mis-handled the gap;
  both now fail loud. (1) `HJBSemiLagrangianSolver` raises when a mixed per-axis BC has segments mapping
  to different geometric operations (e.g. no-flux + periodic) — the characteristic fold and ADI
  diffusion collapse it to the first segment's single op applied to all axes (order-sensitive). (2)
  `HJBFDMSolver.build_linearized_operator` (the strict-adjoint FP operator, #707) raises on any non-
  Neumann/no-flux BC — it hardcodes no-flux at every boundary while the solver declares DIRICHLET for
  the normal solve, so a Dirichlet outflow was silently treated as mass-conserving. Per-axis / adjoint-
  Dirichlet support remains a follow-up.
