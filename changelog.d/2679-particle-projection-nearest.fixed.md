- **A particle that steps into an obstacle always lands navigable, and at the nearest navigable point wherever the
  domain's φ is the true distance to the navigable set, outside a 1e-6 band about a medial axis** (#2679 stage 1,
  #2687; the rest is #2688 and #2689). φ is not that distance for a point trapped against the outer box, inside a
  union or overlapping obstacles, or for an approximate SDF.
  - `FPParticleSolver`'s forward loops place a particle that stepped into an obstacle with
    `ImplicitDomain.project_to_interior`'s projection, now through `ImplicitDomain.place_step` (#2707). They used
    `project_to_domain`, which walked the ray toward the bounding-box centre to the first navigable point. A
    particle that entered the side of an obstacle facing away from the centre came out on the far side -- every
    particle, for an obstacle on the centre.
  - `ImplicitDomain.project_to_interior` no longer returns a point outside the domain. This changes it for every
    caller that reaches it with an `ImplicitDomain`: `ImplicitDomain.place_step` (#2707), the deprecated
    `apply_boundary_conditions("reflecting")`, and in `geometry/boundary/applicator_meshfree.py`
    `ParticleReflector.reflect` (`reflection_type="simple"`) and `ParticleReflector._elastic_reflect`.
    (`MeshfreeApplicator._apply_reflecting_bc` falls back to it only for a geometry without bounds, which an
    `ImplicitDomain` never is: #2693.)
    - It projects onto the boundary along the SDF normal and steps inside by `tolerance`, or by up to
      1e4 * `tolerance` where the projection converges linearly and stops short (an obstacle's corner, the seam of
      a union).
    - A point still outside is placed by the face its step crossed when `previous` is given, and refused by name
      without it (#2688). That is where the domain's φ is not the true distance to the navigable set: a point trapped
      against the outer box, where an obstacle crosses or is flush with it, and a point inside a union, overlapping
      obstacles or an approximate SDF (#2689).
    - Before, it returned the points its projection did not place outside the domain, without a warning.
  - `ImplicitDomain.project_to_domain` is removed; use `project_to_interior`. It was a second projection beside the
    protocol's: the bounding-box ray, a private Newton iteration used only when the ray failed, and a
    `method="gradient"` that its signature and its error message advertised and that raised "Unknown projection
    method".
  - The deprecated `ImplicitDomain.apply_boundary_conditions("reflecting")` now projects with `project_to_interior`,
    so the positions it returns change: a point outside is placed as `project_to_interior` places it, where it moved
    along the ray toward the bounding-box centre.
