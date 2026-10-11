- **The legacy 1-D FDM boundary-condition class is removed:** `mfgarchon.geometry.boundary.fdm_bc_1d.BoundaryConditions`,
  also reachable as `BoundaryConditions1DFDM` and `LegacyBoundaryConditions1D`. It was deprecated in 0.14.0 with a
  runtime warning (#869), and its docstring promised removal in v1.0.0. This user ruling (2026-10-11) supersedes
  that promise and removes it in 0.22.0. Build a `mfgarchon.geometry.boundary.BoundaryConditions` instead.
  `periodic_bc`, `dirichlet_bc`, `neumann_bc`, `no_flux_bc` and `robin_bc` build one condition on every wall. For a
  different value at each wall, which `left_value` and `right_value` gave, use one `BCSegment` per face:
  `BoundaryConditions(dimension=1, segments=[BCSegment(name="left", bc_type=BCType.DIRICHLET, value=1.0,
  boundary="x_min"), BCSegment(name="right", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max")])`. Stored on
  the grid, a BC is shared, and the FP reads its Dirichlet walls as absorbing exits, m = 0; to impose m = g, give
  the FP solver the BC as its own `boundary_conditions`.
  - **Reading either name from `mfgarchon.geometry.boundary`** raises an `AttributeError` that names the
    replacement. A `from mfgarchon.geometry.boundary import ...` of either, and an import of `fdm_bc_1d`, raise
    Python's own `ImportError`.
  - **Every branch that read the class goes with it.** These include the FDM applicator's calculator route and
    ghost update, the interpolation applicator, `get_ghost_values_nd`, FP-FDM's boundary reads, and HJB-GFDM's
    #2513 check of `left_value`/`right_value`. That retires two defects of the legacy route: it imposed
    `left_value` at both walls, and several readers took `left_value` raw.
  - **A BC object FP-FDM cannot honour still fails loud (#1559).** Its time stepping and `build_advection_operator`
    raise a `TypeError` for an object without `is_uniform`; its 2-D tensor-diffusion path refuses one too, with an
    `AttributeError` or a `TypeError`. Before, a legacy `"neumann"` or `"no_flux"` object was assembled as a no-flux
    wall; it is refused too. The interpolation applicator refuses any object that is not a `BoundaryConditions`.
  - **FP-FDM's Dirichlet value at a 1-D wall** is read without a fallback. A value whose evaluation raises an
    `AttributeError` was imposed as 0.0; the error now reaches the caller.
  - **`TensorProductGrid.get_boundary_handler(bc_type)`** is reached only after `grid.boundary_conditions = None`,
    since a grid requires a BC at construction. It now builds the unified class in every dimension, by one factory.
    - On a 1-D grid, `"no_flux"` was a legacy Neumann wall and is `NO_FLUX` now, the zero total flux, and
      `"neumann"`, which raised a `ValueError`, builds `NEUMANN`.
    - Where that default is reached, `custom_conditions` is refused in every dimension; with a stored BC it is
      ignored, as before. On a 1-D grid it built the legacy class, and only for a `bc_type` the grid did not
      recognise; otherwise, and on every grid of two or more dimensions, it was ignored.
    - FP-FDM does not reach this method on a cleared grid: it reads the grid's `get_boundary_conditions()` first,
      whose default is a no-flux BC, and solves that.
