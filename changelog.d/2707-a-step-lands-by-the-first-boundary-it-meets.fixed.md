- **A particle step lands by the first boundary it meets** (#2707, #2679 stage 2b).
  - `ImplicitDomain.place_step(previous, end, via=None)` is the one owner of where a particle step lands inside an
    implicit domain. A path that stays inside lands at its end. An end inside an obstacle lands at its nearest point
    when the path from where it entered the obstacle stays in it and the outward normals at the entry and at that
    point agree; otherwise it lands on the face its path met first. A step past an obstacle's middle, or one that
    jumped over a thin obstacle, used to land on the obstacle's far side; it now lands on the face it entered.
  - `FPParticleSolver` cuts each step 1e-9 of the box's length short of its first wall hit. A step that meets the
    domain's boundary before that is placed by `place_step`, and the outer boundary condition does not act on it, so
    a step that met an obstacle before an absorbing wall is no longer absorbed. The other steps reach the boundary
    condition, and its reflected survivors are placed along their reflected paths.
  - `fp_particle_bc.project_to_navigable_or_defer` is removed. It had no caller outside `FPParticleSolver`, and
    `ImplicitDomain.place_step` is the owner now.
