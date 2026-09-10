- **A boundary datum the FEM assembly cannot see is refused rather than dropped** (Issue #2294).
  `assemble_robin_terms` reads `bc.segments`, so an inhomogeneous Neumann arriving through the
  `default_bc` / `default_value` fall-through was accepted and discarded — the Issue #1686 hole.
  It now raises. The "is this datum verifiably zero?" predicate is delegated to
  `bc_utils.describe_inhomogeneous_bc_data`, the repository's single owner, so a homogeneous wall
  is still a no-op whatever its dtype (`np.float32(0.0)`, `np.int64(0)`, a zero array), while a
  callable- or provider-valued Neumann is now refused instead of silently treated as `g = 0`.
