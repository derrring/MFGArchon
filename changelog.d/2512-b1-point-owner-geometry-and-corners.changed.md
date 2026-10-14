- **The point owner reads its faces from the geometry, and resolves a corner at the point** (#2512 row B1, ahead of
  stage B1c).
  - `BoundaryConditions.segment_for_point` on a BC with no `domain_bounds` reads the faces from the bounds of the
    geometry it is given (or the grid the BC is bound to). It used to refuse with "Either domain_bounds or
    domain_sdf must be set", even when the geometry had bounds. Without either it refuses, naming `geometry=`.
    The face owner decides a `region`'s cover on the same box when there is no grid.
  - A point at a corner takes each face's own segment, read at the point only where that face is partly covered,
    and the corner order chooses (user ruling, 2026-10-14). It took the node's answer before, refusals included,
    so a corner of a face split by `region` into strips with no `default_bc` was refused. Wherever the node
    answers, the point still gives the node's segment.
