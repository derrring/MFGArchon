- **The deprecation guide no longer tells a reader to migrate to a name it elsewhere deprecates.**
  `drift_field` is the destination on `FPFDMSolver.solve_fp_system` — replacing `velocity_field`,
  where it means the optimal control $\alpha^*$ — and is simultaneously deprecated in favour of
  `potential_field` on eight other FP solvers, where it means the value function $U$. The guide
  listed all nine rows with nothing marking them as different quantities, so a reader's reasonable
  conclusion (the name is on its way out everywhere) is the wrong one.

  Both parameters exist on both solver families, so the wrong migration is accepted silently rather
  than raising. Measured on a 21-point 1D problem, `sigma = 0.3`, `T = 0.2`, constant optimal
  control `alpha = 1.0`, initial mass centred at 0.3: `drift_field=alpha` transports the centroid
  to 0.5055, `potential_field=alpha` leaves it at 0.3151 — the solver computes
  $-c\nabla\alpha = 0$ for a constant control, so the advection vanishes entirely. A 37.7% error
  in the transported centroid, no exception and no warning.

  The generator now detects the class mechanically — any identifier that is both a deprecated name
  and a recommended replacement — rather than special-casing this one. It emits a section before
  the version listings pairing each side with its owning API and its own migration, and flags every
  affected row inline, including the `velocity_field -> drift_field` row, which is the row that
  makes the name look canonical and which a check keyed only on deprecated names would leave clean.
  A collision is a legitimate transient, so this surfaces it rather than deciding it; the naming
  question itself is #1786. Issues #1043, #1044.
