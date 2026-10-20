- **A `region` or `sdf_region` exit on part of a face now absorbs a particle stepped past it** (#2780).
  `BoundaryConditions.segment_for_point` reads a point at its nearest point of the box for every face it reads it on,
  by every placement field. Before, only a marked `region_name` was read there, and a `region` such as
  `{"x": (1, 1), "y": (0.3, 0.7)}` or an `sdf_region` held no stepped point (x > 1), so `FPParticleSolver` let every
  particle cross it under `default_bc`. Both now absorb as the same part of the face written with `boundary="x_max"`
  does, and a flux limit on such an exit binds.
  - Past a corner the nearest point moves two coordinates: a particle at (1.2, 1.05), read on x_max, is read at
    (1, 1), so an exit on x_max whose y range ends at 1 now holds it.
  - A marked `region_name` read at a point outside the box with no face is refused by name, where it was read at the
    nearest point: which face the point crossed is unknown. Read it on the face it crossed.
  - A refusal of a point read on a face names the point read, its nearest point of the box.
