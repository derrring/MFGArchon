- **Howard's operator builders close the stencil row unconditionally** (Issue #2066, contract in
  #2081). `_build_dlap_from_socp` / `_build_dgrad_central` wrote weights verbatim and never closed
  the row, which was correct only because their names asserted SOCP input: SOCP weights satisfy
  the sum rule by construction, since `build_taylor_matrix_*` sets `A[:, 0] = 1.0` and the
  consistency constraint `e_lap == A.T @ L` has `e_lap[0] == 0.0`, so row 0 of that equality is
  `sum_j L_j == 0`. Once operators may come from elsewhere that guarantee is gone, and
  `TaylorOperator` / `UpwindOperator` weights multiply deviations `u_j - u_i`: written verbatim
  they give a Laplacian wrong by O(1e+2). The closure is mandatory for deviation weights and
  idempotent for sum-rule ones, so it is applied with no branch on provenance, and the builders
  are renamed `_build_dlap_from_weights` / `_build_dgrad_central_from_weights` — the old names
  asserted an input they no longer receive. Rows accumulate rather than assign, because
  `neighbor_indices` is not unique under a periodic geometry.
