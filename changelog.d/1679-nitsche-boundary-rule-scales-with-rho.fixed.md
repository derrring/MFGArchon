- **The Nitsche Dirichlet boundary rule refines with the cloud, not with the domain**
  (Issue #1679). `_segment_quadrature` called `boundary_tensor_gauss` without `n_cells`, taking its
  default of `1`, so the 2-D boundary rule stayed at **24 points from n=11 to n=26** while the volume
  rule — which does pass `n_cells` — grew 3600 → 22500. The integrand is `phi_i phi_j` and
  `phi_i (n . grad phi_j)`, which varies on the MLS support scale `rho`, and `rho` shrinks under
  refinement: boundary points per support radius fell 2.10 → 0.84 and the assembled block's relative
  quadrature error grew 0.549 → 1.608. Both branches now size their cell count off `rho` — **two
  cells per support radius**, so at least `2 * n_gauss` points fall within every `rho` along a face.

  **This is why the 2-D path diverged.** Measured through the shipped routines on the issue's own
  manufactured Poisson (`u = sin(pi x) sin(pi y)`, `D = 0.5`, `gamma = 100`, `rho = 3.5h`, degree-2
  MLS), the Gauss ladder goes from EOC **-0.49 / -0.93 / -0.84** to **+3.50 / +3.66 / +3.75** and
  SCNI from **+0.39 / -1.69 / -3.08** to **+2.02 / +1.99 / +1.99**. `cond(A)` stops growing
  (4.5e3 → 1.26e4 becomes 4.2e3 → 4.9e3) because `lambda_max` does (1.42e1 → 3.38e1 becomes
  1.33e1 → 1.33e1).

  The guarantee is a **bound**, not a constant: points per support radius lie in
  `[2*n_gauss, 2*n_gauss + n_gauss*rho/max_side)`. It reads 12.60 at `n = 11, 16, 21, 26` only
  because `ceil(4(n-1)/7)` happens to equal `0.6(n-1)` at those values; at n=7 it is 14.00 and at
  n=13, 12.25. The factor of two is where accuracy plateaus, not a copied constant — at n=21 the
  error is 6.7375e-05 at one cell per radius, 5.9721e-05 at two, and 5.9726e-05 at both four and
  eight.

  **The issue's own "sharpest lead" is demoted, not retired.** SCNI's `lambda_min` still collapses
  after the fix (7.33e-05 → 2.25e-06, in fact *steeper*: 20.9x → 32.6x over the ladder) while SCNI
  converges at the optimal +2.00 — so the rank behaviour of the smoothed gradients is not what made
  refinement worse. It is not thereby healthy: SCNI's `cond(A)` still runs like `h^-3.8` against the
  `h^-2` of a well-conditioned second-order operator, and SCNI remains worse than Gauss (11x at
  n=11, 49x at n=26). That half of #1679 survives this change.

  It also explains the non-monotonicity in `n_gauss` the issue recorded as "not a floor": with the
  cell count pinned at 1, raising `n_gauss` at fixed cost cannot keep up with a support scale that
  keeps shrinking — it pushes the turn out rather than removing it.

  Deliberately conservative for an anisotropic box: `max_side` is the largest side of the bounding
  box while `boundary_tensor_gauss` applies `n_cells` per face. It never under-resolves, but a
  1 x 0.01 box over-refines its short faces; making that per-face needs a per-face `n_cells`.

  The curved-boundary (`sdf_region`) path is bit-unchanged (`max(16, max(1, x)) == max(16, x)`,
  verified over 36 configurations), and 1-D is bit-unchanged too — `quadrature.py` builds a 1-D face
  as a single point of unit surface measure on a branch that never reads `n_cells`, which is why the
  1-D EOC test could never see this.
