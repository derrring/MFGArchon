- **A particle at or past a corner of the box takes the owner's corner condition** (#2512 row B1, stage B1d).
  `ParticleApplicator.apply` and `apply_with_flux_limits` named the first face they found a particle on, x before y,
  and read only that face, so the owner's corner order (priority, then DIRICHLET before a natural condition) never saw
  a corner. They now read each boundary particle at its nearest point of the step's box through the contact loop's
  reader, `segment_at_box_contact` (#2733), and `_get_boundary_id` is removed.
  - The particle solver reaches `apply` on its periodic path only: a BC that declares a periodic face and is read per
    point (`needs_segment_aware_bc`), as one with an exit is. There, with x periodic and an exit on y, a particle past
    a corner the two share was wrapped; it is now absorbed, as the corner order gives. Away from a corner a particle
    takes its face's condition, as before.
  - Under `corner_strategy="average"` or `"mollify"`, a corner whose faces impose different conditions is refused, as
    it is for the face families, naming `corner_strategy="priority"`. The first-face read wrapped such a particle in
    silence.
  - Every query reads on the step's box, as the contact loop does. A BC that carries its own `domain_bounds` was read
    on those; it is now read on the box the step was taken in.
  - Row B1 is met, with one listed exception: the deprecated `BaseOptimizationSolver.get_domain_constraints`, which has
    no library caller and is removed by v0.25.0.
