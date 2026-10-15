- **A particle that steps into an obstacle moves to the nearest navigable point, unless it is trapped against the
  outer box; it always lands navigable** (#2679, #2687).
  - `project_to_navigable_or_defer`, which both `FPParticleSolver` forward loops use, now calls
    `ImplicitDomain.project_to_interior`. It used `project_to_domain`, which walked the ray toward the bounding-box
    centre to the first navigable point. A particle that entered the side of an obstacle facing away from the centre
    came out on the far side -- every particle, for an obstacle on the centre.
  - `ImplicitDomain.project_to_interior` no longer returns a point outside the domain. This changes it for every
    caller: `project_to_navigable_or_defer`, the deprecated `apply_boundary_conditions("reflecting")`, and in
    `geometry/boundary/applicator_meshfree.py` `MeshfreeApplicator._apply_reflecting_bc` (its fallback for a
    geometry without bounds), `ParticleReflector.reflect` (`reflection_type="simple"`) and
    `ParticleReflector._elastic_reflect`.
    - It projects onto the boundary along the SDF normal and steps inside by `tolerance`, or by up to
      1e4 * `tolerance` where the projection converges linearly and stops short (an obstacle's corner, the seam of
      a union).
    - A point still outside is walked toward the bounding-box centre to the first navigable point: navigable, not
      the nearest, and the walk can cross the obstacle (#2688). This is reached by an obstacle that crosses or is
      flush with the bounding box, whose nearest face lies outside it. If the walk fails too, it raises
      `RuntimeError` naming the point.
    - Before, it returned those points outside the domain, on the box wall inside the obstacle, without a warning.
  - `ImplicitDomain.project_to_domain` is removed; use `project_to_interior`. It was a second projection beside the
    protocol's: the bounding-box ray, a private Newton iteration used only when the ray failed, and a
    `method="gradient"` that its signature and its error message advertised and that raised "Unknown projection
    method".
  - The deprecated `ImplicitDomain.apply_boundary_conditions("reflecting")` now projects with `project_to_interior`,
    so the positions it returns change: a point outside moves to the nearest boundary point and just inside, where
    it moved along the ray toward the bounding-box centre.
