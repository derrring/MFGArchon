- **The joint-SOCP stencil scale `h_i` is pinned on both dispatch paths, at production's cone
  setting** (Issue #1793). `h_i` non-dimensionalises the monotonicity cone AND appears in the ratio
  reported about it, so inflating it tightens the constraint, shrinks the gradient weights by the
  same factor, and leaves `kappa` inside the bound looking healthy. A 25-axis mutation sweep found
  the suite blind: 5770 passed with `median` replaced by `max`.

  **The two paths behave oppositely, and a pin on one says nothing about the other.**

  On `socp_clarabel`, `kappa_max` is unusable. Where the cone binds it is `C` by construction: at
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
  exactly. Measured, `kappa_max / h_i` agrees to 2e-16 across all four scales, the smallest
  off-centre weight is 3.45, and the value is bit-identical under injected solver tolerance because
  there is no solver. `hjb_gfdm.py:1078` picked `C = 8.0` precisely to land here, so this is the
  path production usually takes.

  Both discriminators verified red against `median -> min`, `-> mean` and `-> max`. Filed #2113 for
  the production consequence: one field name returns qualitatively different things on the two
  paths, and a caller cannot tell which without checking `via`.
