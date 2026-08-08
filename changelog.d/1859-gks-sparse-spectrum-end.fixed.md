`check_gks_stability` no longer reports every operator stable once `N > 100`. The sparse branch
asked `eigs` for `which="LM"` (largest magnitude) while the parabolic criterion is on the largest
real part; for a discretized Laplacian the largest-magnitude eigenvalues are the most negative
ones, so the near-zero eigenvalues that decide stability were never returned. An operator with
`max Re(λ) = +0.1` was reported stable at N = 101, 201 and 401, and correctly unstable at N = 61
only because that took the dense path. The end of the spectrum sampled is now chosen by the
criterion being applied. Sparse-branch tests added — every pre-existing test in the file used
N = 50 or N = 5, so that branch had never been executed. (#1859)
