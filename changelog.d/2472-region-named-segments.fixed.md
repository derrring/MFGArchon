- **A region-named BC segment governs the faces its region covers, and no others** (Issue #2472). With `mark_region("outlet", boundary="x_max")` and `mixed_bc_from_regions`:
  - one region was read as uniform, and its condition was written on every face;
  - with two regions, `get_bc_type_at_boundary` gave the first region every face, while in d >= 2 the FDM ghosts gave it none. The ghost path's region lookup indexed the grid's flat region mask with a face index, raised, and swallowed the error.

  The fix:
  - On a structured grid, `mixed_bc_from_regions` now resolves a region that covers whole faces to those faces when it builds the BC. The solvers and the FDM operators, which read BCs without the geometry, apply it directly.
  - A face counts as covered when the region holds its whole interior; on a two-point axis, its whole face.
  - A region covering part of a face keeps its `region_name`. The face-level readers (`get_bc_type_at_boundary`, the FDM ghost path, `FDMApplicator.enforce_values`) share one resolver, `region_name_governs_face`, which uses the geometry's mask when given the geometry and a region name that is a face label otherwise.
  - **Anything else is refused,** where it was stretched to every face or dropped.
  - `mixed_bc_from_regions` also no longer removes the `"default"` entry from the caller's dict.
- **A uniform BC's unused `default_bc` no longer makes it mixed** (Issue #2472). `geometric_operations` counted the default even when one unrestricted segment covers every face. One periodic segment over a NO_FLUX default was therefore refused by `HJBSemiLagrangianSolver` and `FPSLSolver` as a mixed per-axis BC.
