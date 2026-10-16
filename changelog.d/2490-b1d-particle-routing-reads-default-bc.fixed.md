- **`FPParticleSolver` chooses its fast path by each condition's particle action, `default_bc` included** (#2490
  stage B1d). The routing read the segments alone: a reflecting segment, a placed one, or none, beside
  `default_bc=DIRICHLET` took the reflecting fast path, so the walls the default makes absorbing reflected every
  particle. The fast path, one reflect or wrap on every wall, is now taken where `BoundaryConditions.uniform_segment`
  is one condition everywhere that does not absorb, or where the conditions differ, no segment is placed on part of a
  face, and every condition in play reflects a particle: each segment's, and `default_bc`'s whenever it is set, even
  where the face segments leave it no wall. Every other BC routes per point. Changes from the old routing:
  - Four NO_FLUX faces beside a NO_FLUX default, and two REFLECTING faces with no default, take the fast path, so
    they now run on the grid-drift path with `preserve_indices=True`, which refused them.
  - An absorbing `default_bc` beside reflecting segments routes per point, so the grid-drift path with
    `preserve_indices=True` and the GPU path now refuse it (#1119), where they returned reflected particles.
  - One PERIODIC face beside a default of another condition, an axis periodic on one face only, routes per point,
    where a particle crossing it wraps; the fast path reflected it. Neither answer is supported: #2713 owns that BC,
    and the grid-drift path refuses it on either route.
  - `M_particles_trajectory` is an array on the fast path and a list on the per-point path, so its type changes
    wherever the route does.
  - `needs_segment_aware_bc` returns True for a uniform `robin_bc` with `beta=0`, which absorbs, and raises for an
    EXTRAPOLATION condition; `FPParticleSolver` refuses both at construction.
  The GPU path's refusal names those reasons; it said every BC it refuses needs particles removed at a DIRICHLET
  segment.
