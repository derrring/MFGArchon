- **The joint-SOCP stencil scale `h_i` is pinned at production's cone setting** (Issue #1793). `h_i`
  non-dimensionalises the monotonicity cone AND appears in the ratio reported about it, so
  inflating it tightens the constraint, shrinks the gradient weights by the same factor, and leaves
  `kappa` inside the bound looking healthy. A 25-axis mutation sweep found the suite blind: 5770
  passed with `median` replaced by `max`.

  **`kappa_max` — the only scale-carrying quantity the stencil dataclass exposes — cannot pin the
  scale at any `C`, and the two conditions that would make it usable are mutually exclusive.**
  Where the cone binds it is `C` by construction: at production's `C = 8.0` all four candidate
  scales agree to five decimal places. Where the cone is slack nothing binds, so the objective
  drives an x-axis weight onto the `eps_pos = 0.0` bound and the argmax edge is the one the
  optimiser deleted — `L_1 = 9.88e-08` against `||D_1|| = 4.76e-07`, a `0/0` ratio that tracks
  solver tolerance one-for-one and becomes `inf` at tol `1e-14`. Slack *is* the condition under
  which an edge gets zeroed, so scanning `C` from 1.0 to 8.0 finds no regime with both. Filed as
  #2113.

  The pin therefore reads the constraint rather than the diagnostic: at `C = 8.0` the cone binds on
  one edge, so that edge's weight is what `||D_1|| <= (C / h_i) * L_1` forces. It reads
  3.65e-02 / 6.46e-02 / 7.50e-02 / 9.77e-02 under min / median / mean / max — separations of
  43.4%, 16.2% and 51.3% against 9.6e-05 of drift across solver tolerances 1e-6 to 1e-14. Verified
  red against `median -> min`, `-> mean` and `-> max`, each turning exactly one test red and only
  that one.

  This leaves the diagnostic path unpinned, which is a real gap and not closable with the current
  dataclass. It is recorded in the test rather than papered over.
