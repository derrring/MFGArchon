- **The Nitsche Dirichlet boundary rule refines with the cloud, not with the domain**
  (Issue #1679). `_segment_quadrature` called `boundary_tensor_gauss` without `n_cells`, taking its
  default of `1`, so the 2-D boundary rule stayed at **24 points from n=11 to n=26** while the volume
  rule — which does pass `n_cells` — grew 3600 → 22500. The integrand is `phi_i phi_j` and
  `phi_i (n . grad phi_j)`, which varies on the MLS support scale `rho`, and `rho` shrinks under
  refinement: boundary points per support radius fell 2.10 → 0.84 and the assembled block's relative
  quadrature error grew 0.549 → 1.608. Both branches of the function now size their cell count off
  `rho`, which is what the curved-boundary branch eleven lines above was already doing.

  **This is why the 2-D path diverged.** Measured through the shipped routines on the issue's own
  manufactured Poisson (`u = sin(pi x) sin(pi y)`, `gamma = 100`, `rho = 3.5h`, degree-2 MLS), the
  Gauss ladder goes from EOC **-0.49 / -0.93 / -0.84** to **+3.50 / +3.66 / +3.75** and SCNI from
  **+0.39 / -1.69 / -3.08** to **+2.02 / +1.99 / +1.99**. `cond(A)` stops growing (4.5e3 → 1.26e4
  becomes 4.2e3 → 4.9e3) because `lambda_max` does (1.42e1 → 3.38e1 becomes 1.33e1 → 1.33e1). The
  rule is now 144 → 360 points, holding 12.6 boundary points per support radius at every level.

  **The issue's own "sharpest lead" was not the cause, and this retires it.** SCNI's `lambda_min`
  still collapses after the fix (7.33e-05 → 2.25e-06 over the same ladder) while SCNI converges at
  the optimal +2.00, so the rank behaviour of the smoothed gradients is not what made the answer
  worse under refinement. SCNI remaining *worse than Gauss* — 11x at n=11, 49x at n=26 — survives
  the fix and is a separate question.

  It also explains the non-monotonicity in `n_gauss` the issue recorded as "not a floor": with the
  cell count pinned at 1, raising `n_gauss` cannot converge, because a single cell's rule does not
  resolve a support scale that keeps shrinking.

  The curved-boundary (`sdf_region`) path is bit-unchanged: its `max(16, ...)` floor is preserved.
  1-D is bit-unchanged too — `quadrature.py` builds a 1-D face as a single point of unit surface
  measure on a branch that never reads `n_cells`, which is why the 1-D EOC test could never see this.
