- **A particle step lands by the first boundary it meets** (#2707, #2679 stage 2b).
  - `ImplicitDomain.place_step(previous, end, via=None)` is the one owner of where a particle step lands inside an
    implicit domain. A path that stays inside lands at its end. An end in an obstacle lands at its nearest point when,
    read on the primitive whose surface the path entered (`ImplicitDomain.crossing_leaf`), the path stays inside it
    from the entry to the end and its outward normals at the entry and at that point agree; otherwise it lands on
    the face its path met first. A step past an obstacle's middle, one that jumped over a thin obstacle, one that
    entered an obstacle meeting a wall and ended beyond the wall, and one through a union's concave corner, used to
    land on the far side; each now lands on the face it entered.
  - `ImplicitDomain.crossing_leaf` is the corner owner's one answer to which primitive's surface a crossing is, and
    `crossing_normal` derives from it. A component whose phi is exactly 0 at the crossing's outer point now counts as
    crossed. `project_to_interior` treats a projection whose normal vanishes -- on the internal seam of two touching
    components -- as not placed: it stayed at phi = 0, inside the closed obstacle.
  - `FPParticleSolver` cuts each step 1e-9 of the box's length short of its first wall hit. A step that meets the
    domain's boundary before that is placed by `place_step`, and the outer boundary condition does not act on it, so
    a step that met an obstacle before an absorbing wall is no longer absorbed. The other steps reach the boundary
    condition, and its reflected survivors -- every step that left the box, one that starts on the wall included --
    are placed along their reflected paths.
  - `fp_particle_bc.project_to_navigable_or_defer` is removed. It had no caller outside `FPParticleSolver`, and
    `ImplicitDomain.place_step` is the owner now.
