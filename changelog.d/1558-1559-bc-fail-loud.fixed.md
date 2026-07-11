- **Three silent-wrong boundary-condition paths now fail loud** (Issues #1558, #1559; all off
  published numerics). (1) `SDFParticleBCHandler._compute_normal` raised a `RuntimeError` instead of
  fabricating an arbitrary `[1,0,...]` outward normal when the finite-difference SDF gradient vanishes
  — the fabricated normal reflected particles along a geometry-independent direction (mirrors the
  #1047 `project_to_domain` raise). (2) `TensorProductGrid.get_boundary_handler(bc_type)` now raises
  `ValueError` on an unrecognized `bc_type` instead of silently defaulting to periodic (1D) / neumann
  (nD); the docstring's advertised `periodic_x`/`periodic_both`/`mixed` keys were never implemented in
  the factory and silently became the default — the docstring is corrected to the keys actually
  supported. (3) The FP-FDM time-stepping assembly now raises `NotImplementedError` for a legacy
  `fdm_bc_1d` dirichlet/robin BC instead of silently assembling it as no-flux (legacy periodic/neumann/
  no_flux still assemble; `_is_dirichlet_at_point` cannot see a legacy BC, so a legacy dirichlet was
  silently coerced). Each carries a mutation-verified discriminating test.
