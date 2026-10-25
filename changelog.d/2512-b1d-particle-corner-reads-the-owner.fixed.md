- **A particle at or past a corner of the box takes the owner's corner condition** (#2512 row B1, stage B1d).
  `ParticleApplicator.apply` and `apply_with_flux_limits` named the first face they found a particle on, x before y
  before z, and read only that face, so the owner's corner order (priority, then DIRICHLET before a natural
  condition, then a placed segment before a global one such as `default_bc`, then declaration order) never saw a
  corner. They now read each boundary particle at its nearest point of the step's box through the contact loop's
  reader, `segment_at_box_contact` (#2733), at the owner's default tolerance (1e-8), as the first-face read asked the
  owner.
  `_get_boundary_id` is removed.
  - The particle solver reaches `apply` on its periodic path only: a BC that declares a periodic face and is read per
    point (`needs_segment_aware_bc`), as one with an exit is. There a corner now takes the corner order's condition
    where it took its first face's:
    - x periodic, an exit on y: a particle past their corner was wrapped and is now absorbed;
    - y periodic beside the walls of `default_bc`, whose segment ranks below every declared one: it was reflected and
      is now wrapped, also in 3-D with y or z periodic. The wrap carries it through the wall (#2803);
    - y periodic at a higher priority than an exit or a wall on x: it was absorbed or reflected and is now wrapped;
    - periodic and wall faces at one priority take the one declared first: with x periodic and the walls declared
      first it was wrapped and is now reflected; with y periodic declared first it was reflected and is now wrapped;
    - in 3-D, an exit on z beside a wall on y: it was reflected and is now absorbed.
  - Away from a corner a particle takes its face's condition, as before. One within the owner's face tolerance of a
    second face, 1e-8 plus 1e-12 times that face's bound, is read at their corner, as the owner reads such a point.
  - Under `corner_strategy="average"` or `"mollify"`, a corner whose faces impose different conditions is refused, as
    it is for the face families, naming `corner_strategy="priority"`. This includes conditions a particle cannot tell
    apart, such as NEUMANN 0 beside NO_FLUX, or two exits with different values. Before, the particle took its first
    face's condition.
  - `ParticleApplicator` is exported, and a direct caller sees the corner order for every mix of conditions. Beyond the
    blending refusal, it now also meets four refusals at a corner that the first-face read did not reach: an
    `sdf_region` on a BC not bound to a grid, a `boundary` naming no face of the box, and an unset `default_bc` or a
    ROBIN one beside a face no segment covers. The solver hands the applicator its grid-bound BC.
  - Every query reads on the step's box, as the contact loop does. A BC that carries its own `domain_bounds` was read
    on those; it is now read on the box the step was taken in.
  - Row B1 is met, with one listed exception: the deprecated `BaseOptimizationSolver.get_domain_constraints`, which has
    no library caller and is removed by v0.25.0.
