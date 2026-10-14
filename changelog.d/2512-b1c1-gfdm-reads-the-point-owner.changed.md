- **HJB-GFDM classifies its boundary points through the point owner** (#2512 row B1, stage B1c-1).
  - `HJBGFDMSolver` asks `BoundaryConditions.segment_for_point` for each boundary collocation point, on the problem's
    geometry and at `BOUNDARY_TOL`, with its own equivalence: two segments are the same where GFDM writes the same
    row for them. NEUMANN and NO_FLUX with one datum are one row; `flux_capacity` is not read. Before, it walked the
    segments in its own priority order on the first face it found. A point the owner refuses is refused with the
    owner's reason.
  - At a corner the row is written along the normal of the first face, in face order, whose segment at the point
    writes the row the corner's segment writes. Where the corner's segment changes, the normal follows its datum.
  - `BoundaryConditions.face_segments_at_point` gives each face a point lies on with its segment there: the
    readings `segment_for_point` chooses among.
  - `_infer_boundary_id`, which had no caller, is deleted.
  - The "no `monotonicity_scheme`" warning no longer calls `qp_m_matrix` paper-canonical or M-matrix (#1689).
