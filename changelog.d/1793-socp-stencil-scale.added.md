- **The joint-SOCP stencil scale `h_i` is pinned where the cone has slack** (Issue #1793). `h_i`
  non-dimensionalises the monotonicity cone AND appears in the ratio reported about it, so
  inflating it tightens the constraint, shrinks the gradient weights by the same factor, and leaves
  `kappa` inside the bound looking healthy. A 25-axis mutation sweep found the suite blind: 5770
  passed with `median` replaced by `max`.

  Sharper than "kappa is self-consistent", and measured: **where the cone BINDS, `kappa` is `C` by
  construction and carries zero information about the scale** — at production's `C = 8.0` all four
  candidate scales give `kappa` to within 0.00%. Where it has slack they separate by 3.70%, 1.33%
  and 21.77%. The pin therefore runs at `C = 1.0`, which is a real limit of the approach and is
  stated in the test rather than hidden.

  Two mechanisms: an oracle for the consistency solve (a uniform cross, whose feasible set is a
  single point — so it establishes the Taylor matrix and says nothing about the cone or the scale),
  and the pin. Verified red against `median -> min`, `-> mean` and `-> max`, each turning exactly
  one test red and only that one.
