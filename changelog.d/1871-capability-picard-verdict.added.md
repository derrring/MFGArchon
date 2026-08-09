- **The capability matrix records whether the coupled solve converged** (Issue #1871). Its mass
  oracles could not see it, and not because a tolerance was loose: mass conservation is a property
  of the FP time-stepping, which holds on whatever drift field it is handed, converged or not.
  Measured when this landed, three of the five PASS cells run exactly to their iteration budget
  without converging — `fdm_upwind` and `sl_linear` at 5, `sl_linear_2d` at 3, the last being the
  cell cited throughout #1745 as the control that works where the FDM family fails. In the other
  direction, a deliberately under-relaxed run (`picard.relaxation=0.1`, 3 sweeps) reports
  `converged=False` with mass drift `1.8e-16` and is a **PASS** under the old verdict, so two of the
  UNSUPPORTED cells — `fdm_upwind_2d` and `fvm_muscl_2d` — could have been turned green by a config
  change that leaves the solve further from a fixed point. (`fdm_centered_2d` could not: it raises
  inside the first FP solve, which relaxation of the outer iterate cannot reach.) Every cell holding
  a solver result now carries `picard_converged` and `picard_iterations` in its baseline artifact.
  **Recorded, not gated**: gating today turns three long-standing greens red, and two of those still
  do not converge at 100 sweeps for different reasons — for `fdm_upwind` exactly one of the four
  criterion inputs blocks, the value function's absolute error (rises to 287, ends at `2.3e-04`),
  while for `sl_linear` all four are above tolerance, the density worst and not descending (never
  below `0.357`, ends at `8.2e-01`). That is Issue #1873. No cell changes status.
  Guarded by a source-level test: a cell function that binds a `.solve(...)` result must mention
  `_picard_verdict`, which is structural and so needs no exemption list. An earlier version of that
  test compared recorded artifacts instead and stayed green when a call site was deleted — the
  baseline still carried the field from the last regeneration — which is the same defect class the
  field itself exists to expose, one level up.
