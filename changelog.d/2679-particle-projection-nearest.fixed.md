- **A particle that steps into an obstacle always lands navigable, and at the nearest navigable point wherever the
  obstacle's φ is the true distance in its interior, outside a 1e-6 band about its medial axis** (#2679 stage 1,
  #2687; the rest is #2688 and #2689).
  - `project_to_navigable_or_defer`, which both `FPParticleSolver` forward loops use, now calls
    `ImplicitDomain.project_to_interior`. It used `project_to_domain`, which walked the ray toward the bounding-box
    centre to the first navigable point. A particle that entered the side of an obstacle facing away from the centre
    came out on the far side -- every particle, for an obstacle on the centre.
  - `ImplicitDomain.project_to_interior` no longer returns a point outside the domain. This changes it for every
    caller that reaches it with an `ImplicitDomain`: `project_to_navigable_or_defer`, the deprecated
    `apply_boundary_conditions("reflecting")`, and in `geometry/boundary/applicator_meshfree.py`
    `ParticleReflector.reflect` (`reflection_type="simple"`) and `ParticleReflector._elastic_reflect`.
    (`MeshfreeApplicator._apply_reflecting_bc` falls back to it only for a geometry without bounds, which an
    `ImplicitDomain` never is: #2693.)
    - It projects onto the boundary along the SDF normal and steps inside by `tolerance`, or by up to
      1e4 * `tolerance` where the projection converges linearly and stops short (an obstacle's corner, the seam of
      a union).
    - A point still outside is walked toward the bounding-box centre to the first navigable point. It lands exactly
      where the removed `project_to_domain` put it: navigable, not the nearest, and the walk can cross the obstacle.
      It is reached by a point trapped against the outer box, where an obstacle crosses or is flush with it
      (#2688), and by a point the projection does not converge for inside a union, whose φ is not a true distance
      there (#2689). If the walk fails too, it raises `RuntimeError` naming the point.
    - Before, it returned the points its projection did not place outside the domain, without a warning.
  - `ImplicitDomain.project_to_domain` is removed; use `project_to_interior`. It was a second projection beside the
    protocol's: the bounding-box ray, a private Newton iteration used only when the ray failed, and a
    `method="gradient"` that its signature and its error message advertised and that raised "Unknown projection
    method".
  - The deprecated `ImplicitDomain.apply_boundary_conditions("reflecting")` now projects with `project_to_interior`,
    so the positions it returns change: a point outside moves to the nearest boundary point and just inside, where
    it moved along the ray toward the bounding-box centre.
