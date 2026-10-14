- **One owner decides which boundary segment governs a location, and the FDM family reads it** (#2512, row B1,
  stage B1a; closes #2624 and #2626, and fixes #2490's FDM rows).
  - **The owner.** It is `BoundaryConditions.segment_for_face`, `segment_for_corner` and `segment_for_point`.
    - On a face the order is: priority; then a placed segment before a global one (no placement field, or
      `boundary="all"`, which is every face); then declaration order.
    - A node several faces share takes a choice among those faces' own segments, by one more rule after
      priority: DIRICHLET before any other condition.
    - A point query exactly at a corner that names no face takes each face's own segment, read at the point only
      where a `region` or `sdf_region` covering part of that face is all that keeps the face from answering, and
      the corner order chooses among them. Both steps take the reader's `equivalent=`, so the point gives the
      node's answer wherever the node, asked with the same equivalence, answers. Any other refusal stands: a
      segment shadowed on every face refuses every corner.
      Naming the face (`face=`) reads the point on that face. The particle solver names the first face it finds a
      particle on, so it does not reach this rule.
    - Under `corner_strategy="average"` or `"mollify"`, a corner whose faces impose different conditions is
      refused, and one whose faces agree is answered. Each reader judges "different" by its own sameness: the
      semi-Lagrangian enforcement compares what it writes, so it answers a corner that differs only in
      `flux_capacity`, which the node, the point and the FDM Dirichlet rows refuse.
    - `region` and `sdf_region` are read on the solver's grid, `region_name` on its marks, and
      `normal_direction` selects the face it points out of. Beside an `sdf_region`, a normal picks the face and
      the sdf still decides how much of it the segment holds.
    - **The grid.** A grid binds the BC it stores (`BoundaryConditions.bind_geometry`). When a reader asks without
      a geometry, the owner decides a placement on that grid. The readers that hold none are the ghost pad inside
      an operator, the tensor kernels' no-flux test, the particle applicator and `AdvectionOperator`'s periodic
      span. A reader that holds the grid passes it.
    - Every reader in the FDM family asks it: the HJB-FDM ghosts and Dirichlet rows, the 1-D HJB rows, FP-FDM's
      node classification, the semi-Lagrangian enforcement and per-axis operations, `value_gradient`,
      `get_ghost_values_nd`, and the tensor-diffusion kernels' no-flux test.
    - Before this, each reader had its own rule, and two of them could disagree inside one solve. #2624: a 1-D
      HJB-FDM wall solved as U(0, 0) = 1.025536, which is neither the Dirichlet 3 nor the Neumann wall it was
      given.
  - **Readers outside the FDM family that moved with it.** `get_bc_type_at_boundary`, `get_bc_value_at_boundary`,
    `get_bc_at_point` and `BCSegment.matches_point` now answer by the owner's rules, so the particle solver and the
    GFDM readers that call them changed too:
    - at a point, a placed segment beats a global one of equal priority: a particle reaching an exit on x_max,
      beside a wall declared before it, is absorbed where it was reflected;
    - `normal_direction` selects only the face it points out of; on a box domain it used to match every face;
    - the particle solver's topology query reads a region or sdf placement on the solver's grid, and asks only
      whether an axis wraps.
    - Over the test suite these change no particle or GFDM answer; review-built cases do, and are pinned.
  - **Refused by name, where a reader used to stretch, drop or guess:**
    - a segment covering only part of a face, at a higher priority than the condition on the rest of it, when
      the two differ;
    - a segment that governs no face because something outranks it everywhere, unless it imposes the same
      condition there. Two exits that differ only in `flux_capacity` are different conditions;
    - a segment that reaches no face -- no node of a face's interior, such as a `region` or `sdf_region` between
      the grid's boundary nodes or holding only corner nodes, or a marked region on no face -- unless it imposes
      what governs every face, as the reader compares conditions. The particle topology query, which asks only
      whether an axis wraps, accepts a non-periodic one of these;
    - a face of an axis the domain lacks, refused when the BC is built if its dimension is known, and a `boundary`
      that names no face;
    - a location no segment governs when `default_bc` is unset: "…: no segment governs {where}, and default_bc
      was not specified on BoundaryConditions (#1100). Either cover {where} with a segment -- an unplaced
      segment says 'the rest' -- or pass default_bc= with the condition every location no segment governs
      should get (#2512 row B1).";
    - a ROBIN `default_bc`, which carries no alpha and no beta;
    - an `sdf_region` that returns more than one value for a single point.
  - **Who is affected**, measured over this repository's test suite and counting each BC once per test. These
    readers read 1,072 of the suite's 3,450 BCs. On 28 of those the owner, asked for the whole condition, answers
    a face differently from the old ghost reader:
    - 11 have a face governed by a different segment;
    - 12 are refused by name: 6 partial covers, 2 unset and 2 ROBIN defaults, 1 shadowed segment, and 1
      `boundary` that names no face;
    - 2 carry a door placed by `region` on part of a face. Asked for the whole condition, the owner refuses it as a
      partial cover; the reader that reads these BCs asks only whether an axis wraps, and accepts it;
    - 3 have a face the old reader refused for an unset `default_bc`, which a segment now governs.
  - **API changes.**
    - `BoundaryConditions.uniform_segment` is the segment that governs every face when all faces carry one
      condition: every segment, and `default_bc` where it can govern, imposes the same type, value,
      `flux_capacity` and Robin coefficients. It reads no placement, so it needs no geometry, and it is `None`
      when no segment is global and `default_bc` is unset or ROBIN. Uniform-only fast paths read it instead of
      `segments[0]`.
    - `get_bc_type_string` raises `ValueError` for a BC that `uniform_segment` does not answer: one whose
      segments, and its `default_bc` unless a global segment exists, impose different conditions. It used to
      return the first segment's type. A BC with no segments and a `default_bc` now answers with the default's
      type (#1700 part B).
    - `HJBSemiLagrangianSolver.get_bc_type_for_point` reads the owner per point, and returns "none" off the
      boundary. It used to return `default_bc`'s type for every point.
    - New: `BoundaryConditions.bind_geometry` and `corner_nodes`. `segment_for_corner` takes `equivalent=`.
    - `InterpolationApplicator.enforce_values`, `FDMApplicator.enforce_values`, `get_bc_value_at_boundary`, the
      particle solver's `get_topology_per_dimension`, `apply_diffusion`, `DiffusionOperator`,
      `tensor_calculus.diffusion`, `tensor_calculus.laplacian` and `laplacian_with_bc` take `geometry=`.
    - `needs_segment_aware_bc` refuses a string other than the `"periodic"` sentinel, where it answered False.
      `get_topology_per_dimension` raises `TypeError` for anything but a `BoundaryConditions`, the `"periodic"`
      sentinel or `None`, where it answered bounded walls.
  - **Tensor diffusion (#2626).**
    - A no-flux BC spelt one segment per face no longer leaks mass under an off-diagonal Σ. On #2626's fixture
      it went from −7.042e-03 to −3.928e-17, the uniform spelling's value.
    - No-flux on some walls only, under an off-diagonal Σ, is refused; that capability is #2632.
