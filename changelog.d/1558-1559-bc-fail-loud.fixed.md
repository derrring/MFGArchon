- **Three silent-wrong boundary-condition paths now fail loud** (Issues #1558, #1559; all off
  published numerics). (1) `SDFParticleBCHandler._compute_normal` raised a `RuntimeError` instead of
  fabricating an arbitrary `[1,0,...]` outward normal when the finite-difference SDF gradient vanishes
  — the fabricated normal reflected particles along a geometry-independent direction (mirrors the
  #1047 `project_to_domain` raise). (2) `TensorProductGrid.get_boundary_handler(bc_type)` now raises
  `ValueError` on an unrecognized `bc_type` instead of silently defaulting to periodic (1D) / neumann
  (nD); the docstring's advertised `periodic_x`/`periodic_both`/`mixed` keys were never implemented in
  the factory and silently became the default — the docstring is corrected to the keys actually
  supported. (3) The FP-FDM time-stepping assembly now raises `NotImplementedError` for a legacy
  `fdm_bc_1d` dirichlet/robin/**periodic** BC instead of silently assembling it as no-flux (only legacy
  neumann/no_flux, which ARE no-flux, still assemble). `_is_dirichlet_at_point` cannot see a legacy BC,
  so a legacy dirichlet was silently coerced; and despite the old "relies on no-flux + interior wrapping"
  claim, a legacy `periodic` gets NO wrap here — it is byte-identical to legacy no_flux and differs from
  canonical `periodic_bc` by O(1) once mass reaches the wall (verified with an off-center bump). Each
  carries a mutation-verified discriminating test.
