`ghost_cell_fp_no_flux` no longer carries its own copy of the Robin closed form. The Fokker-Planck
no-flux condition `J . n = 0` is a Robin condition once its axis-frame `v` and `D` are projected onto
the outward normal, and that projection is now `normal_frame_coefficients(v, D, sign) -> (alpha,
beta)`, with the ghost obtained from `ghost_cell_robin`. Verified bit-identical on 12 configurations
— two centrings × two wall directions × three parameter sets — and against the exact zero-flux
profile `rho * exp(v_n*dx/D)`, which neither function implements. (#2128)

One behaviour change, intended: cell-centred at `2D = v_n*dx`, where the ghost's own coefficient
vanishes and the condition determines nothing, now raises instead of silently returning
`interior_value`. The vertex-centred `D ~ 0` degeneracy is a different condition and its pre-existing
value is preserved behind an explicit guard rather than inherited from Robin — that choice is a
physics question and is #2215's, not a consolidation's. (#2128, #2215)
