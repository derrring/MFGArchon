- **The point owner reads its faces from the geometry, and takes the reader's equivalence** (#2512 row B1, ahead of
  stage B1c).
  - `BoundaryConditions.segment_for_point` on a BC with neither `domain_bounds` nor `domain_sdf` reads the faces
    from the bounds of the geometry it is given, or of the grid the BC is bound to. Without such a geometry it
    refuses, naming the geometry it got. A BC's own `domain_sdf` is not overridden by a geometry's bounding box.
    The face owner decides a `region`'s cover on the same box when there is no grid.
  - `segment_for_point` takes `equivalent=`, as `segment_for_face` and `segment_for_corner` do. At a corner it
    decides whether a face answers and which segment the corner takes.
