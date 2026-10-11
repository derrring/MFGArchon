- **The legacy 1-D FDM boundary-condition class is removed:** `mfgarchon.geometry.boundary.fdm_bc_1d.BoundaryConditions`,
  also reachable as `BoundaryConditions1DFDM` and `LegacyBoundaryConditions1D`. It was deprecated in 0.14.0 with a
  runtime warning (#869), and its docstring promised removal in v1.0.0. This user ruling (2026-10-11) supersedes
  that promise and removes it in 0.22.0. Build a
  `mfgarchon.geometry.boundary.BoundaryConditions` instead, with `periodic_bc`, `dirichlet_bc`, `neumann_bc`,
  `no_flux_bc` or `robin_bc`.
  - **The two names** raise an `AttributeError` that names the replacement.
  - **Every branch that read the class goes with it.** These include the FDM applicator's calculator route and
    ghost update, the interpolation applicator, `get_ghost_values_nd`, FP-FDM's boundary reads, and HJB-GFDM's
    #2513 check of `left_value`/`right_value`. That retires two defects of the legacy route: it imposed
    `left_value` at both walls, and several readers took `left_value` raw.
  - **A BC object FP-FDM cannot honour still fails loud (#1559),** now with a `TypeError` for any object that is not
    a `BoundaryConditions`. Before, a legacy `"neumann"` or `"no_flux"` object was assembled as a no-flux wall; it
    is refused too.
  - **`TensorProductGrid.get_boundary_handler(bc_type)` on a 1-D grid** builds the unified class, by the factory
    every dimension uses.
    - Its `"no_flux"` was a legacy Neumann wall and is `NO_FLUX` now, the zero total flux.
    - Its `custom_conditions` is refused. It built the legacy class only for a `bc_type` the grid did not
      recognise, and was ignored otherwise, as it was on every grid of two or more dimensions.
    - The `bc_type` default is reached only after `grid.boundary_conditions = None`, since a grid requires a BC at
      construction. There, FP-FDM's last-resort fallback to the grid's handler used to receive a legacy periodic
      BC, which #1559 refused. It now receives a unified periodic BC, which FP-FDM solves.
