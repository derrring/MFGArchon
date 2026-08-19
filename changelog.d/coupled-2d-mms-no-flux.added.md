A coupled 2D MMS on a no-flux box, run through the production `FixedPointIterator`.

`test_coupled_mfg_mms.py` measures coupled EOC in **1D** under **periodic** BC — deliberately, per
its own note, to avoid the no-flux boundary handling so the measured error is the interior scheme.
So coupled convergence had never been measured in 2D, nor with a wall present at all.

The manufactured pair is the GFDM paper's `eq:mms_reference` / `eq:mms_system` verbatim, on
`Ω = (0,20)²`, `T = 4`, `σ = 1`, `ζ = 1/2`. Sources derived symbolically and cross-checked against a
hand-written vectorised form to `1.1e-16` / `3.3e-19` over 200 random `(t,x)`. Two properties the
paper states are reproduced as independent confirmation of the transcription: `∫_Ω r_m dx = 0`
exactly, and `∂u/∂x_k = ∂m/∂x_k = 0` on every wall — the second is what makes `α·n = 0`, hence
`J·n = 0`, so the pair is exactly compatible with the no-flux wall.

Measured over `Nx = 11/21/31` with `Nt ∝ Nx`: EOC `u` 0.95 → 1.01, `m` 0.99 → 1.00, Picard
converging in 4 outer iterations at every level. 48.9 s, marked `slow` and `integration`.

The test asserts an error **level** beside the order, because an order alone is a min over the HJB
scheme, the FP scheme and the wall closure. Bounds from measurement: baseline `eu` at the finest
level is `5.959e-01`; a solver whose `D` is off by 1.21× gives `1.326e+00` with EOC `u` collapsing
1.01 → 0.38, and by 2× gives `4.465e+00` with EOC → 0.06. Verified mutation-red against the 1.21×
case.

Two limits are stated in the file rather than left to be discovered. The wall is **tangential**
(`∂_ν u = 0`), so this exercises compatibility and not the treatment of a normal drift — that is
#2006, FP-only. And it does not measure the **coupling direction**: `ζm` is ~2.5% of the HJB residual
(RMS `1.28e-03` against `1.24e-01`), which the paper says of its own instance too. A model-side
perturbation of `ζ` is swamped by the ~30% relative discretization error at these resolutions; a
solver-side error is not, which is what the discrimination above measures.
