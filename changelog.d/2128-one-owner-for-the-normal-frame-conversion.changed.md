`ghost_cell_fp_no_flux` no longer carries its own copy of the Robin closed form. The Fokker-Planck
no-flux condition `J . n = 0` is a Robin condition once its axis-frame `v` and `D` are projected onto
the outward normal, and that projection is now `normal_frame_coefficients(v, D, sign) -> (alpha,
beta)`, with the ghost obtained from `ghost_cell_robin`. Verified against the pre-#2128 closed form
over 2016 inputs — 1966 identical, 50 of them the first change below, 0 regressions — and against the
exact zero-flux profile `rho * exp(v_n*dx/D)`, which neither function implements. On the 12 headline
configurations, re-measured on the shipped build: 5 bit-identical, 6 at 1 ulp, and 1 at 3 ulps
(4e-16 relative), all from summation order. (#2128)

Three behaviour changes, all intended and all narrow:

1. Cell-centred at `2D = v_n*dx`, where the ghost's own coefficient vanishes and the condition
   determines nothing, now raises instead of silently returning `interior_value`.
2. Multi-element `diffusion_coeff` now raises `NotImplementedError` on both centrings. Previously it
   raised `ValueError` for a multi-element ndarray and `TypeError` for a list or tuple, and — because
   `bool()` on a one-element array is legal — silently accepted size 1. Size-1 is still accepted; only
   the exception **type** changed, which is a public-API change for a caller catching **either**
   `ValueError` or `TypeError` here.
3. A field-valued `drift_velocity` now works on the cell-centred path, where it previously raised
   `ValueError: truth value of an array is ambiguous`. A gain, not a regression.

The vertex-centred `D ~ 0` degeneracy is a different condition and its pre-existing value is
preserved behind an explicit guard rather than inherited from Robin — that choice is a physics
question and is #2215's, not a consolidation's. (#2128, #2215)
