- **`FPParticleSolver` chooses its fast path by each condition's particle action, `default_bc` included** (#2490
  stage B1d). The routing read the segments alone. A reflecting segment, a placed one, or none, beside
  `default_bc=DIRICHLET` took the reflecting fast path, so the walls the default makes absorbing reflected every
  particle.
  - The fast path applies one reflect or wrap on every wall. It is now taken in two cases. The first is where
    `BoundaryConditions.uniform_segment` is one condition everywhere that does not absorb. The second is where the
    conditions differ, no segment is placed by `region`, `sdf_region`, `region_name` or `normal_direction`, and every
    condition in play reflects a particle: each segment's, and `default_bc`'s whenever it is set, even where the face
    segments leave it no wall. Every other BC routes per point.
  - Changes from the old routing:
    - An absorbing `default_bc` beside reflecting segments, placed or not, routes per point. Its walls now absorb on
      the CPU paths, where they reflected every particle. The GPU path refuses it, where it returned reflected
      particles.
    - Reflecting conditions with a segment placed by more than a face label route per point. The CPU paths reflect
      there as the fast path did, so the answer is unchanged.
    - **The 1-D GPU path now refuses those BCs.** At main it ran them on its fast path, and its particles there
      matched the CPU path's. A placed `region_name="x_min"` beside a NO_FLUX default is one such BC. The GPU path has
      no per-point boundary (#2764): use `backend="numpy"`, or place the segment by face label (`boundary="x_min"`).
      An n-D solve does not take the GPU path, so nothing changes there.
    - Four NO_FLUX faces beside a NO_FLUX default, and two REFLECTING faces with no default, take the fast path, where
      they routed per point. The answer is the same.
    - One PERIODIC face beside a default of another condition, an axis periodic on one face only, routes per point,
      and so does its mirror, a PERIODIC default beside one face of another condition. There a particle crossing it
      wraps, where the fast path reflected it. Neither answer is supported: #2713 owns that BC, and the grid-drift
      path refuses it on either route.
    - A reflecting segment on a `boundary` label that names no face of the box (such as `boundary="inlet"`) beside
      another reflecting condition takes the fast path, whose topology read refuses it by name. The old routing sent
      it per point, where the callable path ran it.
    - In 2-D and 3-D, a reflecting segment placed by a `region_name` the grid marks on part of a face routes per
      point. The old routing sent it to the fast path, whose topology read refused it on every path; the callable path
      now runs it.
    - `M_particles_trajectory` is an array on the fast path and a list on the per-point path, so its type changes
      wherever the route does.
    - `needs_segment_aware_bc` returns True for a uniform `robin_bc` with `beta=0`, which absorbs, and raises for an
      EXTRAPOLATION condition, where it returned False for both. `FPParticleSolver` refuses both at construction.
  - The GPU path's refusal names what routes per point. It said the BC needs particles removed at a DIRICHLET
    segment, which names one of the cases that route per point.
