- **HJB-GFDM classifies its boundary points through the point owner** (#2512 row B1, stage B1c-1).
  - `HJBGFDMSolver` asks `BoundaryConditions.segment_for_point` for each boundary collocation point, on the problem's
    geometry and at `BOUNDARY_TOL`, with its own equivalence: two segments are the same where GFDM writes the same
    row for them. NEUMANN and NO_FLUX with one datum are one row; `flux_capacity` is not read. Before, it walked the
    segments in its own priority order on the first face it found. A point the owner refuses is refused with the
    owner's reason, each point on its own line.
  - Both point readers take a point's faces from one place: the box (`domain_bounds`, or the geometry's bounds when
    the BC has neither bounds nor `domain_sdf`), else the one face the BC's `domain_sdf` gives. A BC with only a
    `domain_sdf` is now classified by its own SDF; one whose SDF is an obstacle's puts the outer walls on no face and
    is refused, naming `domain_bounds`.
  - On a curved `domain_sdf` a mixed BC is now classified, where it was refused, and its rows take the SDF's normal
    at the point, the one the uniform path writes there.
  - At a corner the row is written along the normal of the first face, in face order, whose segment at the point
    writes the row the corner's segment writes. Where the corner's segment changes, the normal follows its datum.
  - `BoundaryConditions.face_segments_at_point` gives each face a point lies on with its segment there and the
    normal a row along it writes: the readings `segment_for_point` chooses among.
  - `_infer_boundary_id`, which had no caller, is deleted.
  - The "no `monotonicity_scheme`" warning no longer calls `qp_m_matrix` paper-canonical or M-matrix (#1689).
