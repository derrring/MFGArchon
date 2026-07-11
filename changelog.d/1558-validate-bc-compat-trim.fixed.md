- **`validate_bc_compatibility` no longer emits a contradictory BC-type verdict** (Issue #1558,
  defect 3). Its per-discretization support table was a second capability source that disagreed with
  the #1456 single source (`solver.supported_bc_types`): it flagged Robin as "limited" for GFDM while
  `hjb_gfdm._SUPPORTED_BC_TYPES` includes Robin, and allowed Robin for FDM while `hjb_fdm` excludes it
  (and read `default_bc`, which is None for segment-based BCs, so for those it silently returned no
  issues). Trimmed to the one real check — BC/geometry dimension compatibility; BC-type support is
  authoritatively enforced at solve time by each solver's `supported_bc_types`. `discretization` is
  kept in the signature for API stability.
