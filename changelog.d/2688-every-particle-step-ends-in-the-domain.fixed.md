- **Every forward step of `FPParticleSolver` with an `implicit_domain=` records its particles inside that domain,
  or refuses** (#2688, #2679 stage 2; the second half of #1561).
  - The problem's geometry, when it is an `ImplicitDomain` other than a `Hyperrectangle` on the grid's box, is read
    as that domain too (#2708).
  - **One end of step.** The 1-D grid-drift, n-D grid-drift and callable-drift paths end each step in one place,
    `_finish_step`.
    - The outer boundary condition runs on the raw step first: absorption at Dirichlet segments, then reflection.
    - Then the survivors that ended inside an obstacle are placed, with their positions before the step.
    - Before, the 1-D grid-drift path never placed them. The n-D paths placed them before the wall reflection,
      which put particles back inside an obstacle that meets the wall.
  - **The face of entry.** A particle that the projection onto the boundary does not place is now placed by the face
    its step crossed. This reverses part of #2687's entry.
    - `ImplicitDomain.project_to_interior(points, previous=..., via=...)` finds the first crossing on the step's
      path and projects the point onto that face's tangent plane. The path is the segment from `previous`, or for a
      reflected step the polyline through its wall hits (`reflection_waypoints`), so the face is one the particle met.
    - The crossing is found by a scan of 32 steps per leg and bisection on the sign of φ. A leg whose path inside a
      feature is shorter than a scan step, 1/32 of the leg, can step over it: a thin feature, or a corner the leg
      clips (#2710).
    - The face's normal is the crossed component's own, decided by which component's φ changes sign across the
      crossing (`ImplicitDomain.crossing_normal`, overridden by the CSG domains). Within 1e-6 of the outer box, the
      domain's gradient normal blended the box wall's into an obstacle face's. Next to a wall, a particle stepping
      onto the face was then either refused or placed on the obstacle's far side.
    - On a flat face this is the projected-Euler (Skorokhod) map, which keeps the displacement along the face. Where
      the result lands in a second face, the point is projected onto the wedge the faces bound. It meets up to 8
      faces, and a point those faces do not place raises, naming the point and the faces.
    - The walk toward the bounding-box centre is deleted. Without `previous`, `project_to_interior` raises at exactly
      the points it used to walk, naming the point and the route. #2687's "always lands navigable" holds only with
      `previous`.
  - **Initial particles are drawn navigable.**
    - `sample_from_density(..., navigable=)` redraws, within its cell, a jitter that lands where `navigable` is
      False.
    - Cells carrying mass with no navigable point on a 5-per-axis probe lattice hold mass outside the domain, inside
      an obstacle or beyond its outer boundary, and no redraw can place it. Their share of the mass, over every cell that carries any, is decided by
      `mass_fabrication_gate.leave_out_or_raise`, against the threshold that bounds a non-negativity clip (1e-8):
      - above it the sample is refused, naming the share;
      - at or below it those cells are left out, silently, as a clip of that size is, and the rest renormalised.
    - The 1-D grid-drift path samples the nodes themselves and decides its nodes outside the domain the same way.
    - `initial_particles` outside the domain are refused.
  - **Refused rather than delivered wrong:**
    - an `implicit_domain` with any periodic face on the outer boundary, since a wrapped step is not the segment
      from where it started (#2704);
    - an `implicit_domain` whose bounding box is not the grid's box. The boundary condition reflects at the grid's
      box and the placement acts inside the domain's: where the grid's box reaches further, a particle between the
      two is placed by neither; where the domain's does, a particle can be placed beyond the reflecting wall;
    - an `implicit_domain` on the GPU path, whose particles never met the placement.
  - **Known limitation,** which predates this change: a step that passes through an obstacle and ends past an
    absorbing exit is absorbed before any placement (#2707).
