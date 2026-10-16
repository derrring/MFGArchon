- **A particle step lands by the first boundary it meets** (#2707, #2679 stage 2b).
  - `ImplicitDomain.place_step(previous, end, via=None)` is the one owner of where a particle step lands inside an
    implicit domain. A path that stays inside lands at its end. A path that leaves lands at the nearest point of the
    primitive whose surface it crossed first (`ImplicitDomain.crossing_leaf`), on the domain's side, when the path
    stays beyond that surface from the crossing to the end and the outward normals there and at that point agree;
    otherwise on the face its path met first. These used to land on a surface the path never met, and now land on
    the one it crossed:
    - a step past an obstacle's middle, or over a thin obstacle;
    - a step into an obstacle that meets a wall, ending beyond the wall;
    - a step through a union's concave corner;
    - a step out of one room of a union domain toward another, or out of a ring's hole;
    - a step past a room's corner, which could also raise.
  - `ImplicitDomain.crossing_leaf` is the corner owner's one answer to which primitive's surface a crossing is, and
    `crossing_normal` derives from it. A component whose phi is exactly 0 at the crossing's outer point now counts as
    crossed.
  - `ImplicitDomain.navigable` is where a particle may start: the closure of the domain's interior. `FPParticleSolver`
    reads it for its initial particles, in place of `contains`. A point where an obstacle meets the wall it is flush
    with, or on the seam of two touching obstacles, is in the closed domain with no interior next to it, and a step
    from it raised; it is now refused at t = 0, by name.
  - `FPParticleSolver` cuts each step 1e-9 of the box's length short of its first wall hit. A step that meets the
    domain's boundary before that is placed by `place_step`, and the outer boundary condition does not act on it, so
    a step that met an obstacle before an absorbing wall is no longer absorbed. The other steps reach the boundary
    condition, and its reflected survivors -- every step that left the box, one that starts on the wall included --
    are placed along their reflected paths.
  - `fp_particle_bc.project_to_navigable_or_defer` is removed. It had no caller outside `FPParticleSolver`, and
    `ImplicitDomain.place_step` is the owner now.
