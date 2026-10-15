- **A particle that steps into an obstacle moves to the nearest navigable point** (#2679).
  - `project_to_navigable_or_defer`, which both `FPParticleSolver` forward loops use, now calls
    `ImplicitDomain.project_to_interior`: onto the obstacle's boundary along its normal, then just inside. It used
    `project_to_domain`, which walked the ray toward the bounding-box centre to the first navigable point. A particle
    that entered the side of an obstacle facing away from the centre came out on the far side -- every particle, for
    an obstacle on the centre.
  - `ImplicitDomain.project_to_domain` is removed; use `project_to_interior`. It was a second projection beside the
    protocol's: the bounding-box ray, a private Newton iteration used only when the ray failed, and a
    `method="gradient"` that its signature and its error message advertised and that raised "Unknown projection
    method".
  - The deprecated `ImplicitDomain.apply_boundary_conditions("reflecting")` now projects with `project_to_interior`.
