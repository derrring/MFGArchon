- **Six dead accessors removed from the deprecated `fdm_bc_1d.BoundaryConditions`** —
  `is_periodic`, `is_dirichlet`, `is_neumann`, `is_no_flux`, `is_robin`, `get_matrix_size`. All six
  have zero attribute-access callers in `mfgarchon/` and `tests/`, confirmed by an AST sweep over
  every `.py` file as well as by grep. The module has been deprecated since v0.14.0 (2025-12-04)
  against a policy window of "3 minor versions OR 6 months" (`AGENTS.md`); the published line is
  v0.21.0, so both arms closed long ago — seven minor versions and about eight months.
  `get_matrix_size` was not a valid convention that had become obsolete: it was self-inconsistent.
  It sized `neumann` as `M+1` and `no_flux` as `M` although no-flux *is* a Neumann-type condition,
  which requires two different griddings inside one method, and it sized `dirichlet` as `M-1`
  against its own parameter docstring ("number of interior grid points"), under which the interior
  points are exactly the Dirichlet unknowns. The one branch that is true — periodic sheds the
  repeated endpoint — already has a live, single-sourced owner in `repeated_endpoint_count` /
  `periodic_axis_span` (Issue #1822), so nothing was lost. Refs #1559, whose evidence cites two of
  the removed names.
