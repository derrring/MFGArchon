- **What a particle does at the boundary has one owner, `particle_action_at`** (#2490 stage B1d, #2512 row B1;
  fixes #1561 item 1).
  - `ParticleApplicator` asks it for every particle at or beyond the box. The owner clamps the particle onto the box
    and reads `BoundaryConditions.segment_for_point` there. The applicator named the first face it found the
    particle on, x before y, and asked that face alone, so `corner_strategy` never saw a corner: a reflecting face
    on an earlier axis swallowed an absorbing face on a later one. A particle beyond a corner now takes the corner's
    segment in the owner's order: priority, then DIRICHLET before a natural condition.
  - The clamp stands in for the point where the path crossed the wall until #2733 passes the exact contact. A
    particle beyond two faces reads their corner even where its path crossed only one. A `region` or `sdf_region`
    placement is read at the clamp, on the wall where it is placed: a particle beyond an exit at
    `region={"x": (1, 1), ...}` was read off the wall and reflected, and is absorbed.
  - Two segments are one wall where they do the same to a particle, so a REFLECTING face beside a NO_FLUX one is
    one wall, and a NO_FLUX strip of higher priority on a REFLECTING face does not take a corner it shares with an
    exit: the face carries its condition, so the corner reads the face.
    The flux-limited applicator also tells exits apart by `flux_capacity`. A `corner_strategy` other than
    `"priority"` does not choose between two actions at a corner: that corner is refused by name, with its remedies.
    An exit on x_max with no-flux walls and `"average"` absorbed at the corner, and is now refused.
  - A periodic axis is identified, not walled: a particle is wrapped on it first, and the owner reads the faces of
    the other axes alone (`segment_for_point(axes=...)`). A particle beyond a periodic face and a wall was wrapped on
    both axes; it wraps on the one and takes the wall's action on the other.
  - The owner reads a particle's faces with the applicator's boundary tolerance, 1e-10, where its default is 1e-8;
    its face test adds |bound| × 1e-12, and its sub-face covers read 1e-8.
  - A box with an axis of no width is refused: a particle there lies on both faces of that axis at once.
  - `FPParticleSolver` reads every vertex of the box at construction, so a refusal the corner rule raises fires
    there, not at the step a particle first reaches the corner. That is a limit, not coverage: in 3-D two faces meet
    along the box's 12 edges, and a segment restricted by region can change which two govern along an edge, so a
    refusal inside an edge still fires when a particle first reaches it. A BC with no `domain_bounds` is given the
    solver's box first, as the applicator gave it at its first step.
  - Off a grid, an `sdf_region` segment is refused at construction: its cover of a face is decided on a grid's
    nodes, so a particle at a corner could not be read. On main the region was read at the particle beyond the wall
    and the solve ran. Solve on a TensorProductGrid, or place the segment with `boundary=` and `region=`.
  - `FPParticleSolver` routes a BC to the per-point path unless every wall takes one non-absorbing particle action,
    by `BoundaryConditions.uniform_segment_under` the particle's sameness. The routing counted segments and read no
    `default_bc`, so a reflecting segment beside `default_bc=DIRICHLET` took the reflecting fast path, and the walls
    the default makes absorbing reflected. Whole-face segments with no `default_bc` stay on the per-point path, as on
    main: their walls' action is not read without their placements. The GPU path's refusal names those reasons; it
    said every such BC needed particles removed at a DIRICHLET segment.
  - A BC handed to `FPParticleSolver` is completed with the grid's periodic node layout, as every other route
    completes it (#1822), and bound to the grid, as the grid binds its own: an `sdf_region` or `region_name`
    segment is read on it, and a `region` segment's cover of a face is read on the grid's nodes rather than on the
    box. An exit at `region={"x": (1, 1), "y": (0.05, 0.95)}` holds every interior node of x_max, so it governs the
    face and its corner: a particle beyond (1, 1) reflected on main and is absorbed.
  - `ParticleApplicator._get_boundary_id` is removed; nothing calls it.
