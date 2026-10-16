- **What a particle does at the boundary has one owner, `particle_action_at`** (#2490 stage B1d, #2512 row B1;
  fixes #1561 item 1).
  - `ParticleApplicator` asks it for every particle at or beyond the box. The owner clamps the particle onto the box
    and reads `BoundaryConditions.segment_for_point` there. The applicator named the first face it found the
    particle on, x before y, and asked that face alone, so `corner_strategy` never saw a corner: a reflecting face
    on an earlier axis swallowed an absorbing face on a later one. A particle beyond a corner now takes the corner's
    segment in the owner's order: priority, then DIRICHLET before a natural condition.
  - The clamp stands in for the point where the path crossed the wall until #2733 passes the exact contact. A
    particle beyond two faces reads their corner even where its path crossed only one.
  - `FPParticleSolver` routes a BC to the per-point path unless `BoundaryConditions.uniform_segment` names one
    non-absorbing condition. The routing counted segments and read no `default_bc`, so a reflecting segment beside
    `default_bc=DIRICHLET` took the reflecting fast path, and the walls the default makes absorbing reflected.
  - A BC handed to `FPParticleSolver` is completed with the grid's periodic node layout, as every other route
    completes it (#1822).
  - `ParticleApplicator._get_boundary_id` is removed; nothing calls it.
