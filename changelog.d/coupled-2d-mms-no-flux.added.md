A coupled 2D MMS on a no-flux box, run through the production `FixedPointIterator`.

`test_coupled_mfg_mms.py` measures coupled EOC in **1D** under **periodic** BC — deliberately, per
its own note, to avoid the no-flux boundary handling so the measured error is the interior scheme.
So coupled convergence had never been measured in 2D, nor with a wall present at all.

The manufactured pair is the GFDM paper's `eq:mms_reference` / `eq:mms_system` with **two deliberate
changes**, on `Ω = (0,20)²`, `T = 4`, `σ = 1`, `ζ = 1/2`. It is not the paper's pair verbatim, and
both changes exist because the verbatim one cannot express the defects this test is for:

- `c = π/L` instead of `2π/L`. Still Neumann-compatible (`sin(cL) = 0` either way) but no longer
  L-periodic, so the mirror ghost and the periodic ghost stop coinciding. With the paper's `c`,
  swapping the entire boundary-condition family to `periodic_bc` left every assertion green; with
  this one it moves `eu` from `3.0125e-01` to `1.2939e+01` and collapses EOC `u` to `−0.118 / 0.067`.
- `β = 0.6` breaking the symmetry between the axes, plus a `2c/4c` mix in `m`. The paper's pair is
  exactly transpose-symmetric, so `eu` from `U[0]` and from `U[0].T` is bit-identical and a
  wrong-axis read is unexpressible; here the transposed read is 45.7× / 81.8× / 119.5× off.

Sources re-derived for **this** pair with sympy and cross-checked against the hand-written
vectorised forms to `5.6e-17` / `1.1e-19` over 400 random `(t,x)`, against residual scales `2.3e-01`
/ `4.5e-04`, with a positive control (a `1e-7` relative perturbation registers at `4.5e-11`).

Two properties the paper states also hold for this pair — `∫_Ω r_m dx = 0` exactly, and
`∂u/∂x_k = ∂m/∂x_k = 0` on all four walls. **Neither is a check on the transcription**: the first is
one scalar per time and is blind to anything that integrates away, the second never touches the
sources. The second matters for a different reason — it makes `α·n = 0`, hence `J·n = 0`, so the pair
is exactly compatible with the no-flux wall, which is what lets it run on this box.

Measured over `Nx = 11/21/31` with `Nt ∝ Nx`: `eu` `3.0125e-01 → 1.5759e-01 → 1.0534e-01`, EOC `u`
0.935 → 0.993; `em` `9.876e-03 → 5.036e-03 → 3.383e-03`, EOC `m` 0.972 → 0.982. Picard converges in
4 outer iterations at every level, and at every `σ` tried (1.0, 0.5, 0.3, 0.1). 47.5 s, marked
`slow` and `integration`.

Three limits are stated in the file rather than left to be discovered.

**It cannot resolve a coefficient error below ~10%.** Measured across two solver-side defect
families: a 5% error in the diffusion coefficient or in the drift scale passes every assertion. The
error-level assertion is kept as a regression guard on the constant — EOC is a ratio and is blind to
a uniform scaling — but it is *not* a discriminator: over ten measured mutants it never fails
without the EOC assertion failing first.

**The wall is tangential** (`∂_ν u = 0`), so this exercises compatibility and not the treatment of a
normal drift — that is #2006, FP-only.

**It does not measure the coupling direction**, which the paper says of its own instance too. `ζm`
has RMS `1.26e-03` — 8.1% of the `|∇u|²/2` term and 1.1% of the whole HJB residual, which is
dominated by `−∂_t u`. The discretization error at the finest level is 0.41% relative in `u` and
6.6% in `m`, so a model-side perturbation of `ζ` is not resolvable here.

Separately, the study established which solvers can run an MMS at all on this box: **2 of 8 solver
pairings**. `FDM × FDM` and `GFDM × FDM` converge; the rest never enter the solve because
`solve_*_system` does not take a `source_term` (#1991, #2020) or crash on the source argument
convention (#2019).
