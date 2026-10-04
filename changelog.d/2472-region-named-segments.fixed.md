- **A region-named BC segment governs the faces its region covers, and no others** (Issue #2472). With `mark_region("outlet", boundary="x_max")` and `mixed_bc_from_regions`:
  - one region was read as uniform, so its condition was written on every face;
  - with two regions, `get_bc_type_at_boundary` gave the first region every face, while the FDM ghosts gave it none. The ghost path's region lookup indexed the grid's flat region mask with a face index, raised every time, and swallowed the error.

  One resolver, `region_name_governs_face`, now serves both readers. It uses the geometry's region mask when the geometry defines the region, and a region name that is a face label (`"x_max"`, `"left"`) otherwise.
  - A region governs a face only if it covers the whole face. A region covering part of a face is refused, because one condition per face cannot represent it.
  - A reader with no geometry, such as `get_bc_type_at_boundary`, refuses a region name it cannot resolve instead of giving it every face. **Behaviour change:** a solver that types faces through that accessor now raises on a region-named BC whose names are not face labels, where it previously used the wrong condition.
- **A uniform BC's unused `default_bc` no longer makes it mixed** (Issue #2472). `geometric_operations` counted the default even when one unrestricted segment covers every face. One periodic segment over a NO_FLUX default was therefore refused by `HJBSemiLagrangianSolver` and `FPSLSolver` as a mixed per-axis BC.
