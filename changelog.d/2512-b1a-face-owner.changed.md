- **One owner decides which boundary segment governs a location, and the FDM family reads it** (#2512, row B1,
  stage B1a; closes #2490, #2624, #2626).
  - **The owner.** It is `BoundaryConditions.segment_for_face`, `segment_for_corner` and `segment_for_point`.
    - On a face the order is: priority; then a placed segment before a global one (no placement field, or
      `boundary="all"`, which is every face); then declaration order.
    - At a corner one more rule comes after priority: DIRICHLET before any other condition. A point at a corner
      and a node at it give one answer.
    - `region` and `sdf_region` are read on the solver's grid, `region_name` on its marks, and
      `normal_direction` selects the face it points out of.
    - Every reader in the FDM family asks it: the HJB-FDM ghosts and Dirichlet rows, the 1-D HJB rows, FP-FDM's
      node classification, the semi-Lagrangian enforcement and per-axis operations, `value_gradient`,
      `get_ghost_values_nd`, and the tensor-diffusion kernels' no-flux test.
    - Before this, each reader had its own rule, and two of them could disagree inside one solve. #2624: a 1-D
      HJB-FDM wall solved as U(0, 0) = 1.025536, which is neither the Dirichlet 3 nor the Neumann wall it was
      given.
  - **Refused by name, where a reader used to stretch, drop or guess:**
    - a segment covering only part of a face, at a higher priority than the condition on the rest of it, when
      the two differ;
    - a segment that governs no face because something outranks it everywhere, unless it imposes the same
      condition there. Two exits that differ only in `flux_capacity` are different conditions;
    - a face of an axis the domain lacks, refused when the BC is built if its dimension is known, and a `boundary`
      that names no face;
    - a location no segment governs when `default_bc` is unset: "…: no segment governs {where}, and default_bc
      was not specified on BoundaryConditions (#1100). Either cover {where} with a segment -- an unplaced
      segment says 'the rest' -- or pass default_bc= with the condition every location no segment governs
      should get (#2512 row B1).";
    - a ROBIN `default_bc`, which carries no alpha and no beta.
  - **Who is affected.** Measured over the test suite's own 25,489 BC constructions, outside this change's own tests:
    - 56 that an old reader resolved are now refused: 19 for a shadowed segment, 16 for a ROBIN `default_bc`, 10
      for a `boundary` that names no face, 8 for a partial cover and 3 for an unset `default_bc`;
    - 19 resolve a face to a different segment than the old ghost reader gave it;
    - 19 more use `region` or `sdf_region` and are decided only on the solver's grid.
    - The 1-D HJB-FDM hybrid of #2624 is now refused, naming the outranked segment.
  - **API changes.**
    - `BoundaryConditions.uniform_segment` is the segment that governs every face when all faces carry one
      condition: every segment, and `default_bc` where it can govern, imposes the same type, value,
      `flux_capacity` and Robin coefficients. It reads no placement, so it needs no geometry, and it is `None`
      when no segment is global and `default_bc` is unset or ROBIN. Uniform-only fast paths read it instead of
      `segments[0]`.
    - `get_bc_type_string` raises `ValueError` for a BC whose faces differ. It used to return the first
      segment's type. A BC with no segments and a `default_bc` now answers with the default's type (#1700 part
      B).
    - `HJBSemiLagrangianSolver.get_bc_type_for_point` reads the owner per point, and returns "none" off the
      boundary. It used to return `default_bc`'s type for every point.
    - `InterpolationApplicator.enforce_values` and `FDMApplicator.enforce_values` take `geometry=`.
  - **Tensor diffusion (#2626).**
    - A no-flux BC spelt one segment per face no longer leaks mass under an off-diagonal Σ. On #2626's fixture
      it went from −7.042e-03 to −3.928e-17, the uniform spelling's value.
    - No-flux on some walls only, under an off-diagonal Σ, is refused; that capability is #2632.
