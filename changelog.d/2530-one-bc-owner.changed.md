- **One owner decides which BC a problem's solvers share** (#2512 row B9; closes #2530 and #2589).
  - **The owner** is `MFGProblem.get_boundary_conditions`. A solver handed its own BC uses that one. Everything
    else asks the owner: the solvers' BC lookup, FP-FDM, FP-FVM, the particle solver, and the gradient operators
    of HJB-FDM, HJB-SL and FP-SL, which used to be built on the grid's own BC whatever the solver had resolved.
  - **Before this, each reader had its own order.** FP-FDM and FP-FVM read `components.boundary_conditions`
    first, while the HJB solvers, FP-SL and the particle solver read the geometry's. On #2530's route
    (`spatial_bounds=` with a BC in components), with a 1-D exit of 0.7:
    - FP-FDM absorbed: mass 19 → 11.06;
    - HJB-FDM, HJB-GFDM, HJB-SL, FP-SL and the particle solver imposed no-flux;
    - FP-FVM refused the Dirichlet.
  - **The owner's order:**
    - **The geometry's BC, when the geometry declares one.** A grid always does, and a mesh does when a BC is
      attached (`UnstructuredMesh.has_explicit_boundary_conditions` is now True then).
      - A components BC given as well must declare the same condition (`BoundaryConditions.declares_the_same_as`).
        Otherwise both are refused by name: "MFGProblem.get_boundary_conditions: the problem declares its boundary
        conditions twice, …".
      - On the `spatial_bounds=` route the grid is built with the components BC, so there the two are one.
    - **The components BC**, on a geometry that declares none: an implicit domain, a point cloud, or a mesh with
      none attached.
    - **A graph geometry's node BCs**, which are its own (#1471). They are returned, None included, and a
      components BC on a network is refused.
    - **The geometry's own default**, where it has one.
    - **`periodic_bc`**, for a geometry periodic on every axis.
    - **Otherwise a refusal by name, citing #2589.** A mesh with no BC used to reach the Picard loop and fail there
      with `AttributeError: 'NoneType' object has no attribute 'has_providers'`.
  - **Removed fallbacks.** Each live caller was counted over the test suite at 79fae339, before removal:
    - the implicit `periodic_bc` default for a problem whose geometry declares no BC, deprecated since 2026-01-11
      (0 callers);
    - FP-FDM's and FP-FVM's `no_flux_bc` fallbacks (0 callers each);
    - FP-FDM's `get_boundary_handler()` step (0 callers);
    - the particle solver's `"periodic"` string for a torus (3 tests, which now receive `periodic_bc`).

    `get_topology_per_dimension` and `needs_segment_aware_bc` now refuse a string.
  - **Inside `using_resolved_bc`**, a components BC is resolved together with the geometry's, so the two
    declarations stay one.
  - **Who is affected**, measured over this repository's test suite at 79fae339, before the change.
    - **Unchanged:** the owner gives the very object each reader used to pick:
      - FP-FDM: 601 constructions, in 349 tests;
      - FP-FVM: 80, in 45;
      - the particle solver: 116;
      - the solvers' BC lookup: 480,018 queries.
    - **Changed:**
      - two tests declared a components BC that differs from the grid's, one through FP-FDM and one asking the
        owner directly. Both are now refused.
      - two gave FP-FDM and FP-FVM a components BC equal to the grid's. They now read the grid's, the same
        condition.
      - three torus tests now receive `periodic_bc`.
    - **Test doubles that adapt:** four HJB-GFDM tests whose mock problem had no BC owner, and one particle test
      that passed a bare `Mock` as its problem.
