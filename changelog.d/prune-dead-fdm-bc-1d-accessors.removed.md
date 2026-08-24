- **Six dead accessors removed from the deprecated `fdm_bc_1d.BoundaryConditions`** —
  `is_periodic`, `is_dirichlet`, `is_neumann`, `is_no_flux`, `is_robin`, `get_matrix_size`. All six
  have zero attribute-access callers in `mfgarchon/` and `tests/`, and the module has been
  deprecated since v0.14.0 against a policy window of "3 minor versions OR 6 months"
  (`AGENTS.md`); the published line is v0.21.0, so the window closed four minor versions ago.
  `get_matrix_size` additionally described a formulation the package no longer uses — it sized the
  system by eliminating boundary unknowns (dirichlet `M-1`, neumann/robin `M+1`), while the current
  solvers size on the full grid and impose boundary conditions by row replacement and ghost cells.
  `num_interior_points` appears nowhere else in the package, so there was no live counterpart to
  relocate the convention to.
