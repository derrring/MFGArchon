- **The joint-SOCP stencil scale `h_i` is pinned on both dispatch paths, at production's cone
  setting** (Issue #1793). `h_i` non-dimensionalises the monotonicity cone AND appears in the ratio
  reported about it, so inflating it tightens the constraint, shrinks the gradient weights by the
  same factor, and leaves `kappa` inside the bound looking healthy. A 25-axis mutation sweep found
  the suite blind: 5770 passed with `median` replaced by `max`.

  **What decides whether `kappa_max` is usable is not which path ran, but WHY the fast path was
  skipped.** Three exits reach the SOCP and only two force anything: a cone rejection puts the
  optimum on the cone, so `kappa_max = C`; an M-matrix rejection leaves a positivity bound active,
  so the argmax is `0/0`. The third is `np.linalg.solve` raising on a singular `AᵀA`
  (`joint_socp.py:246`), where neither check ever ran and nothing is forced. Both fixtures in this
  file take that exit. Two earlier versions of this changelog asserted the degeneracy of every `C`
  and then of every solver-path stencil; both were false, the second refuted by this file's own
  uniform-cross oracle, where the cone is slack at 0.005 against `C = 1.0` with every off-centre
  weight at 100.0.

  On `SCALE_STENCIL`, `kappa_max` is unusable. Where the cone binds it is `C` by construction: at
  `C = 8.0` all four candidate scales agree to five significant figures. Where the cone is slack
  nothing binds, so the objective drives an x-axis weight onto the `eps_pos = 0.0` bound and the
  argmax edge is the one the optimiser deleted — `L_1 = 9.88e-08` against `||D_1|| = 4.76e-07`, a
  `0/0` ratio that tracks solver tolerance one-for-one and becomes `inf` at tol `1e-14`. So the pin
  there reads the **constraint** instead: at `C = 8.0` the cone binds on one edge, and that edge's
  weight is what `||D_1|| <= (C / h_i) * L_1` forces. It reads 3.65e-02 / 6.46e-02 / 7.50e-02 /
  9.77e-02 under min / median / mean / max. It is a function of `C / h_i` alone — `(C=8, h=0.1)`
  and `(C=16, h=0.2)` give the same `L_1` to nine digits.

  On `wendland_lsq_fast_path` no solver runs: the Wendland least-squares weights are accepted if
  they already satisfy the cone, so `(L, D)` do not depend on `h_i` and `kappa_max = h_i * const`
  exactly. Measured, `kappa_max / h_i` agrees to 1 ulp across all four scales -- stated as a bound
  rather than an exactness, because "spread exactly 0.0" was true at one value of the fixture's
  `delta` multiplier and false at the one actually shipped. The smallest off-centre weight is
  3.4558, and the value does not move under injected solver tolerance because there is no solver.
  `hjb_gfdm.py:1078` picked `C = 8.0` partly to reach
  this path; how often it is reached is not measured and is not asserted.

  Both discriminators verified red against `median -> min`, `-> mean` and `-> max`. Filed #2113 for
  the production consequence: one field name returns qualitatively different things on the two
  paths, and a caller cannot tell which without checking `via`.
