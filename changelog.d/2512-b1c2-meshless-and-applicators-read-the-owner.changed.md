- **Meshless Galerkin's Nitsche path, the meshfree and implicit applicators, and FEM's shared nodes read the BC owner**
  (#2512 row B1, stage B1c-2; #2490, #1947).
  - **Nitsche** asks `BoundaryConditions.segment_for_face` which segment governs each bounding-box face, and imposes
    Dirichlet data on the faces a Dirichlet condition governs. Its own face reader, `nitsche._segment_faces`, is gone.
    - A segment placed by `region`, `sdf_region`, `normal_direction` or `region_name` now governs the faces it covers
      whole; it was refused. A partial cover is refused by the owner.
    - A DIRICHLET `default_bc` is imposed, with `default_value`, on the faces it governs; it was refused.
    - A face no segment governs, with no `default_bc`, is refused; it got the natural condition. Pass
      `default_bc=NO_FLUX` for the condition it was getting.
    - A segment that governs no face, because others outrank it on every face it names, is refused; the
      higher-ranked segment alone was solved.
  - **`MeshfreeApplicator.apply`, `apply_particles` and `ImplicitApplicator.apply`** impose one condition at every
    boundary point, so they read the BC's one condition for every location, `BoundaryConditions.uniform_segment`:
    its type, its value and its Robin coefficients. They read the type from `default_bc` and the value from
    `default_value`. A BC with no such condition is refused by name (#1947); resolving each point's condition is #2712.
    So a segment with no placement now imposes its own condition where `default_bc` named another, and a BC whose
    conditions differ by location is refused where it was flattened onto every point.
  - **FEM** asks the owner to choose a node's segment only where a DIRICHLET segment shares the node. A natural
    condition is imposed on facets, not nodes, so under `corner_strategy="average"` or `"mollify"` a node only natural
    segments share is no longer refused.
