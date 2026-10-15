- **Every forward step of `FPParticleSolver` records its particles inside an obstacle domain, or refuses** (#2688,
  #2679 stage 2; the second half of #1561).
  - The 1-D grid-drift, n-D grid-drift and callable-drift paths end each step in one place, `_finish_step`. The
    outer boundary condition runs on the raw step first (absorption at Dirichlet segments, then reflection). Then the
    survivors that ended inside an obstacle are placed, with their positions before the step.
    - Before, the 1-D grid-drift path never placed them.
    - The n-D paths placed them before the wall reflection, which put particles back inside an obstacle that meets
      the wall.
  - A particle that the projection onto the boundary does not place is now placed by the face its step crossed.
    This reverses part of #2687's entry. `ImplicitDomain.project_to_interior(points, previous=...)` finds the first
    crossing on the segment from `previous` and projects the point onto that face's tangent plane.
    - On a flat face this is the projected-Euler (Skorokhod) map: it keeps the displacement along the face, and it
      cannot tunnel.
    - Where that lands in a second face, the point is projected onto the wedge the faces bound. It meets up to 8
      faces, and a point those faces do not place raises, naming the point and the faces.
    - The walk toward the bounding-box centre is deleted. It placed these points before, and it could cross the
      obstacle.
    - `project_to_interior` without `previous` now raises at exactly those points, naming the point and the route.
      #2687's "always lands navigable" holds only with `previous`.
  - Initial particles are drawn navigable.
    - `sample_from_density(..., navigable=)` redraws, within its cell, a jitter that lands where `navigable` is
      False.
    - A sampled cell with no navigable point is refused, naming the share of the mass it carries: that mass is
      inside an obstacle, and moving it would change the caller's measure.
    - The 1-D grid-drift path samples the nodes themselves, so it refuses a sampled node inside an obstacle.
    - `initial_particles` inside an obstacle are refused.
  - Refused rather than delivered wrong:
    - an `implicit_domain` with a periodic outer boundary, since a wrapped step is not the segment from where it
      started (#2704);
    - an `implicit_domain` on the GPU path, whose particles never met the placement.
  - Known limitation, which predates this change: a step that passes through an obstacle and ends past an absorbing
    exit is absorbed before any placement.
