- **The FEM boundary-load assembly delegates "is this datum zero?" to its single owner** (Issue
  #2294). `bc_utils.describe_inhomogeneous_bc_data` now answers it, so a homogeneous Neumann wall is
  a no-op whatever its dtype (`np.float32(0.0)`, `np.int64(0)`, a zero array), where an
  `isinstance(g, (int, float))` check refused those. A callable- or provider-valued Neumann is
  refused rather than silently read as `g = 0`, and a non-finite datum is refused rather than
  assembling a NaN load. A datum attached only to the `default_bc` fall-through is loaded on the
  facets no segment holds (#2305, closed by #2512's stage B1b).
