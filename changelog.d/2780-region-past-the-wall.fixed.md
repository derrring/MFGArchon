- **A `region` or `sdf_region` exit on part of a face now absorbs a particle stepped past it as its `boundary=` twin
  does** (#2780). `BoundaryConditions.segment_for_point` reads a point at its nearest point of the box for every face
  it reads it on, by every placement region (`region`, `sdf_region`, `region_name`). Before, only a marked
  `region_name` was read there: a `region` such as `{"x": (1, 1), "y": (0.3, 0.7)}` held no stepped point (x > 1),
  and an `sdf_region` only the stepped points inside it, so `FPParticleSolver` let the others cross under
  `default_bc`. Both now absorb as the same part of the face written with `boundary="x_max"` does, and a flux limit on
  such an exit binds.
  - Past a corner the nearest point moves two coordinates, and a corner particle is not read right in general,
    before or now: the applicator receives only the stepped point, so the face it crossed is unknown. A particle at
    (1.2, 1.05), read on x_max, is now read at (1, 1), so an exit on x_max whose y range ends at 1 absorbs it. That is
    right for one that crossed x = 1 first (from (0.9, 0.95) it crosses at y = 0.983), and wrong for one that crossed a
    reflecting y_max first: (1.05, 1.2), reached from (0.95, 0.9) across y = 1 at x = 0.983, is now absorbed at the
    x_max exit. #2490's corner reading and #2733's crossing point decide which.
  - A collocation point of HJB-GFDM within its boundary tolerance outside a face is read at the face, so an
    `sdf_region` whose edge crosses the face beside it can now hold it.
  - A marked `region_name` read at a point outside the box with no face is refused by name, where it was read at the
    nearest point: no face was passed to read it on. Read it on the face it crossed.
  - A refusal of a point read on a face names the point read, its nearest point of the box.
  - The outward normal a `normal_direction` reads under `domain_sdf` is read at the point as asked, as before.
