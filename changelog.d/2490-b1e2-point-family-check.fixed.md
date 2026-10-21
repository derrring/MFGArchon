- **A point reader refuses, by name, a segment that governs none of its points** (#2490 stage B1e-2, #2512 row B1).
  `BoundaryConditions.refuse_segments_governing_no_point` is the face owner's shadow rule over a reader's own points:
  a segment that governs none of them is refused unless it imposes the same condition, in that reader's sameness, as
  what outranks it where it is held -- off a corner the reading there, at a corner that face's own segment, which the
  corner rule chooses among. The refusal names the segment, where it is held and what outranks it there.
  - `FPParticleSolver` runs it at construction, on the BC it solves with (passed to it, or the problem's), whenever
    the per-point path is taken. It reads one point in each cell of an arrangement of each face
    (`BoundaryConditions.face_cell_centres`), through the applicator's own calls: where a particle crosses a face.
    Before, an exit under a higher-priority wall, a `region` or `sdf_region` off the box, or a `normal_direction` no
    face points along was dropped with no word, and every particle stayed. Two segments are the same to a particle
    when they take one action and neither carries a `flux_capacity` (`same_particle_action`); a segment's value is
    never read.
  - Refused for particles with a reason of its own, though it catches particles today, only because a particle is
    read where its step ends rather than where it crossed (#2733):
    - a set of no width on a face -- a `region` meeting the face only at a corner or an edge, or in a range that is
      one coordinate -- which a crossing particle meets with probability zero; an exit at the corner (1, 1) absorbed
      all 200 particles driven into it at f2dcb164. Give it a range of positive width, or place it by `boundary=`;
    - a `region` lying where the BC's `domain_bounds` reach past the solver's box: an exit at x = (1.05, 1.2) of a
      BC bounded at x = 1.2 absorbed 200 of 200 at f2dcb164. Give the BC the solver's box and place the exit on its
      faces, or place it on the solver's face by `boundary=`;
    - a `normal_direction` no face's normal lies within its `normal_tolerance` of, read under a box `domain_sdf` past
      a corner: (1, 1)/sqrt(2) at 0.9 absorbed 199 of 200 at f2dcb164. Point it along a face's normal, or lower the
      tolerance.
  - An `sdf_region`, or a `normal_direction` read from `domain_sdf`, is read on a lattice of at most
    `FACE_LATTICE_CELLS` = 1024 cells per face, so a set one holds that contains no box of side 1/n of the face's
    extent (1/1024 of an edge in 2-D, 1/32 in 3-D) is refused, the lattice named.
  - Any refusal the owner gives at one of the points the check reads, one in each cell of each face -- an unset
    `default_bc` there, a marked `region_name` whose grid does not reach it -- is raised at construction, where before
    the first particle to reach that point raised it.
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
- **A `region` key has one reader, `region_axis`** (#2490 B1e-2): a name (x, y, z, w) or an integer, a numpy integer
  included, naming one of the domain's axes; `BoundaryConditions.validate` reports any other key by the same rule. The
  face readers now read a numpy-integer key, which they refused. A float key, a negative index, a bool and an unknown
  name are refused by every reader, where the point reader truncated a float to an axis, counted a negative index from
  the last axis and read `True` as axis 1.
