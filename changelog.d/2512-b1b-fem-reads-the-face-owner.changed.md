- **The FEM solvers read the face owner** (#2512, row B1, stage B1b; closes #2595 and #2305). `HJBFEMSolver` and
  `FPFEMSolver` decided which boundary facets a segment governs by a rule of their own: each segment's `boundary` name
  looked up in the mesh's tags, the whole boundary for no name, no priority, no other placement field, and no
  `default_bc`.
  - **The owner's answer per facet** (`BoundaryConditions.segments_for_facets`). The face owner's order and sameness
    rule (#2512 B1a), applied to each boundary facet:
    - priority, then a placed segment before a global one, then declaration order. A NO_FLUX wall named `x_min`
      beside a DIRICHLET segment with no `boundary` is a wall; it was condensed to the Dirichlet value;
    - `region` and `sdf_region` are read at the facet's midpoint, as the point owner reads them, and
      `normal_direction` at its outward normal. A region holding part of a face's facets governs those facets: the
      face family's partial-cover refusal does not apply to a reader whose unit is the facet. These fields used to be
      ignored, so a region-placed exit condensed the whole boundary;
    - `region_name` is the mesh tag of that name;
    - a facet no segment holds takes `default_bc` with its `default_value`. A DIRICHLET default is imposed and a
      NEUMANN(g) default is loaded (#2305); both used to be dropped, the facet getting the natural condition.
  - **Names.** A `boundary` name that is a tag of the mesh is that tag, so a gmsh group called `"left"` is that group.
    Only a name that is no tag is read as the box face it aliases (`parse_boundary_face`): `"left"` is the `x_min`
    tag, `"right"` the `x_max` tag. `"all"` is every facet.
  - **A node shared by facets that different segments govern** takes the corner rule
    (`BoundaryConditions.segment_for_shared_node`): priority, then DIRICHLET before any other condition, then placed
    before global, then declaration order. **Two Dirichlet values at a shared FEM node now take the corner rule
    instead of being refused**: between two placed segments of equal priority, the one declared first holds the node.
    They used to be refused on the condensation path and taken from the last segment declared on HJB-FEM's Newton path
    (#2541). Where a
    higher-priority natural segment meets a Dirichlet facet, the node is the natural segment's and is left free; it
    used to be condensed.
  - **Refused by name**, at construction and at every read of the BC:
    - a name that is neither a tag of the mesh nor a box face the mesh tags, naming both readings: a falsy `0` or
      `""` used to be read as no name, the whole boundary;
    - **a facet that falls through to a `default_bc` that is unset**, which was solved with the natural condition: an
      FEM BC that names some faces and leaves the rest without a default is now refused,
      "BoundaryConditions.segments_for_facets: no segment governs {n} boundary facet(s) of the mesh, and default_bc
      was not specified on BoundaryConditions (#1100). Either cover {n} boundary facet(s) of the mesh with a segment
      -- an unplaced segment says 'the rest' -- or pass default_bc= with the condition every location no segment
      governs should get (#2512 row B1)." Pass
      `default_bc=NO_FLUX` for the condition those facets were getting;
    - a facet that falls through to a ROBIN `default_bc`, which carries no alpha and no beta;
    - a segment that governs no facet, because something outranks it everywhere it reaches or it reaches none, unless
      it imposes the same condition there. A segment on an axis tag that holds no facet, as on a disc, is one; the
      refusal names the tag and lists the tags that hold facets;
    - a DIRICHLET segment that governs facets but whose every node the corner rule gives to segments imposing a
      different condition, such as an exit on one facet between two higher-priority walls. P1 FEM imposed no exit there.
      If a segment that takes one of its nodes imposes its own condition, that condition holds at that node, and it is
      not refused;
    - a tag that also holds facets inside the domain, which main condensed;
    - under `corner_strategy="average"` or `"mollify"`, a node a DIRICHLET segment shares with a segment imposing a
      different condition. A node only natural segments share needs no choice, since a natural condition is imposed on
      facets (#2512 B1c-2).
  - **API changes.** `fem.bc_adapter.is_pure_neumann(mesh, bc)` and `refuse_what_fem_cannot_impose(basis, bc,
    consumer)` take the mesh and the basis. New: `BoundaryConditions.segments_for_facets`, `segment_for_shared_node`,
    `refuse_a_segment_left_no_node`, `conditions.BoundaryFacets`, and `fem.bc_adapter.has_dirichlet_dofs`.
    `_find_segment_dofs` and `_find_segment_facets` are gone.
  - **The point family's readings** of `region` and `sdf_region` now go through the functions the facet entry uses
    (`region_holds`, `sdf_region_holds`). Two answers of `BCSegment.matches_point` change: a `region` naming an axis
    the point lacks raises even when an earlier axis already excludes the point, where it returned False; and a NaN
    coordinate or SDF value no longer matches.
  - **Not here.** The particle solver reads faces by its own rule until stage B1d; GFDM and meshless Nitsche read
    the owner since stages B1c-1 and B1c-2.
