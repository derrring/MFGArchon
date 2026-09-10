- **The FEM boundary-load assembly reads `bc.segments`, and says so** (Issue #2294). An
  inhomogeneous Neumann arriving only through the `default_bc` / `default_value` fall-through
  reaches no assembly and is dropped. That is pre-existing -- `apply_bc_to_fem_system` has always
  been segments-only -- and is now disclosed at the site and tracked, rather than guarded here: two
  guards were attempted and both were wrong, one rejecting `neumann_bc(g)` (whose factory sets both
  channels in lockstep) and its replacement under-refusing on `boundary="left"`, an alias the raw
  `mesh.boundaries` lookup does not resolve.
  The "is this datum verifiably zero?" predicate is delegated to
  `bc_utils.describe_inhomogeneous_bc_data`, the repository's single owner, so a homogeneous wall is
  a no-op whatever its dtype (`np.float32(0.0)`, `np.int64(0)`, a zero array) while a callable- or
  provider-valued Neumann is refused rather than silently read as `g = 0`.
