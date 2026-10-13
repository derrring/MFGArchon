- **`FPSLSolver` states the norm its density converges in** (closes #2468; the scheme question is #2645).
  - At σ = 0 the density converges in W₁ and L¹, which is what the scheme's theory gives (Carlini–Silva,
    arXiv:1708.02042, Theorem 4.1). It does not converge pointwise at a node where the drift changes sign: the
    linear deposit gives that node twice the continuity equation's inflow, so the peak of a compressive drift stays too
    high under refinement. On #2468's fixture the peak is 4.20 against an exact 3.02 at every resolution.
  - With σ > 0 the excess is diffused and the density converges pointwise.
  - The splatting methods' orders, O(dx³) for cubic and O(dx⁵) for quintic, were stated as the density's accuracy. They
    are the interpolations' orders, and the docstrings now say so. The linear method's "O(dx) accuracy" is replaced by
    the norm statement above.
  - `docs/user/CONVENTIONS.md` § 9 gains the rule: a density converges in a named norm, and a peak value or a plot of
    a σ = 0 FP-SL density is a pointwise reading.
