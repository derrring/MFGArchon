- **A point reader refuses, by name, a segment that governs none of its points** (#2490 stage B1e-2, #2512 row B1).
  `BoundaryConditions.refuse_segments_governing_no_point` is the face owner's shadow rule over a reader's own points:
  a segment that governs none of them is refused unless it imposes the same condition, in that reader's sameness, as
  what governs where it is held. The refusal names the segment, where it is held and what outranks it there.
  - `FPParticleSolver` runs it at construction, on the BC it solves with (passed to it, or the problem's), whenever
    the per-point path is taken. It reads one point in each cell of each face where every placement reads one answer
    (`BoundaryConditions.face_cell_centres`), through the applicator's own calls. Before, an exit under a
    higher-priority wall, a `region` or `sdf_region` off the box, or a `normal_direction` no face points along was
    dropped with no word, and every particle stayed. Two segments are the same to a particle when they take one
    action and neither carries a `flux_capacity` (`same_particle_action`); a segment's value is never read.
  - A set of no width on a face, such as a `region` whose range there is one coordinate, holds no cell, and a particle
    reaches it with probability zero: refused. An `sdf_region`, or a `normal_direction` read from `domain_sdf`, is
    read on a lattice of at most `FACE_LATTICE_CELLS` = 1024 cells per face, so a set one holds that contains no box of
    side 1/1024 of an edge's extent (1/32 of a 3-D face's) is refused too, the lattice named.
  - A BC that leaves part of a face to an unset `default_bc` is refused at construction, with the owner's message,
    where before the first particle to reach that part raised it.
  - `HJBGFDMSolver` runs it on its boundary collocation points once it has classified them, with its own sameness
    (`_same_bc_row`). It refused most such BCs only through its corner points, with "pre-classification failed for
    4/40 boundary points"; it now names the segment. An exit strip under a wider wall strip, which covers part of a
    face ahead of its winner, was admitted with the exit read at no point; it is now refused.
- **A point at a corner no longer runs the face owner's whole-BC check** (#2490 B1e-2). The check reads which segments
  govern a face on a grid's nodes, so HJB-GFDM's admission of a BC depended on whether its points included a corner:
  a boundary point inside an exit strip between the grid's nodes was refused with the corners in the cloud and read
  as the exit without them. Both clouds are now admitted, and read alike. `segment_for_point` and
  `face_segments_at_point` at a corner answer where the face owner refuses the whole BC; `segment_for_face`,
  `segment_for_corner` and the face readers keep the check.
