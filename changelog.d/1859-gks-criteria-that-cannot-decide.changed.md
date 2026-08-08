`check_gks_stability` now refuses the two `pde_type` branches that could not answer their own
question, instead of returning a reassuring verdict.

- **`"hyperbolic"` raises `NotImplementedError`.** The criterion was `|Im(λ)| ≤ 10·max|λ|`, both
  sides reduced from the same eigenvalue array — and `|Im z| ≤ |z|` holds for every complex
  number, so it reported `stable=True` for every input ever constructed, including transport
  operators amplifying by 7.8e42 and `A = 1e6·I`. Over 200 000 random matrices the worst observed
  ratio was 0.9999999996 against a threshold of 10. It is not repaired to a weaker form because
  the intended check is not answerable per operator: boundedness of `exp(t·A)` is invariant under
  `A → c·A`, so no single-operator predicate may reference `dx`, and `‖A‖ = O(1/dx)` already holds
  by consistency. Cross-grid behaviour belongs in `check_gks_convergence`.
- **`"elliptic"` now uses the full spectrum, and raises `ValueError` above `max_dense_size`
  (new parameter, default 2000).** Definiteness is a statement about every eigenvalue; the sparse
  path sampled one end, and the opposite end is where an indefiniteness sits. Measured: the same
  operator returned the correct verdict at N=50 and N=100 and the wrong one at N=101/200/400.

Both branches gain must-reject controls — tests that feed a known-bad operator and assert the
checker says bad. There were none before: `"hyperbolic"` and `"elliptic"` appeared zero times in
`tests/`, which is why a criterion that could never fail went unnoticed. (#1859)
