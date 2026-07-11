- **`MeshfreeApplicator` now reads Robin coefficients from the BCSegment that carries them** (Issue
  #1558, defect 1). `alpha`/`beta` live on `BCSegment`, not on `BoundaryConditions`, so the previous
  `getattr(boundary_conditions, "beta", 0.0)` always read `0.0` — silently collapsing every Robin BC
  to pure Dirichlet: `apply()` forced a hard `u = g/alpha` instead of the penalty blend, and
  `apply_particles()` always absorbed instead of reflecting. Both paths now read `(alpha, beta)` from
  the ROBIN segment via a single-source helper, so a Robin BC with `beta != 0` behaves as a Robin BC.
  Behavior change on the meshfree Robin path only; off published numerics (published adjoint-consistent
  Robin runs through the `hjb_gfdm` row builder, not `MeshfreeApplicator`). Two mutation-verified tests
  (field penalty-blend, particle reflect-not-absorb).
