- **The particle solver's coupling to time is now tested, on both the 1D and the nD path**
  (Issue #1792). The FP sweep runs forward while the HJB sweep runs backward, and that pairing is
  what makes the MFG fixed point mean anything — but reading `U[T - t_n]` instead of `U[t_n]`
  survived a 25-axis mutation sweep against the entire marker-filtered suite (5770 passed, 0
  failed, as reported in the issue). Re-measured directly: with both index sites mirrored, all 23
  test files mentioning `FPParticleSolver` return 322 passed / 13 xfailed — byte for byte the
  unmutated result. Negative indexing keeps every access in range for any `Nt`, so nothing raises and nothing
  warns while the density tracks the control's journey backwards. The gap was structural rather than
  a missing assertion: every `U_solution` the particle tests build is `np.zeros((Nt, Nx))` or
  `np.tile(f(x), (Nt, 1))`, i.e. time-constant, which makes `U[n]` and `U[-1-n]` the same array and
  the mutation a no-op by construction — no assertion added to such a fixture could have helped.
  The new test drives a well whose centre travels 0.2 → 0.8 and measures the correlation between the
  density's centroid and the well's position over the whole trajectory: **+0.9917 (k=2) and +0.9981
  (k=5) correct, against −0.0143 and −0.6194 for the mirrored read**. Both indexing sites are
  covered (`U_solution_for_drift[n_time_idx]` on the 1D path, `[t_idx]` on the nD one); the issue
  named only the first. The under-driven-fixture trap it warns about — where neither run tracks
  anything, the gap is 1%, and a null result reads as a small effect — is an explicit control. That
  control IS a tuned threshold, unlike the correlation: the mirrored read lags 0.168 at k=2 and
  passes it, so the k=2 row is carried by the correlation alone. Determinism comes from the seed added in #1838; without it this comparison
  is a statement about the Monte Carlo draw rather than about the solver.
  Scoped deliberately: the observable detects the DIRECTION of the time coupling, not the exact
  level. A pure shift `U[i+s]` survives for s from −8 to +20 — with finite drift the mass trails a
  moving well, so reading ahead compensates the lag instead of exposing it — and an off-by-one is
  therefore not caught. The white-box spy used on the FDM side would catch it; that seam does not
  exist on the particle path. The docstring says so rather than implying the stronger claim.
