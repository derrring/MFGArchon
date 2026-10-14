- **The SL pair's `diffusion_theta` documents what it trades at a no-flux wall** (#1855). The default does not change:
  per the user's answer of 2026-10-14, the trade-off is the caller's to configure.
  - At a no-flux wall, measured in 1-D on #1855's fixture with dt the step the advection takes, the split step's steady
    state depends on (1 − θ)·dt. The wall node's bias is the semi-discrete steady state's, which is first order in h,
    plus an excess of the same sign of about K·(1 − θ)·|α|·dt/h, with K = 0.74 to 1.02 over both walls and
    nx = 41 to 321. That excess does not refine away along dt ∝ h at θ = 0.5, and is absent at θ = 1. A sub-stepped
    step keeps it near K·(1 − θ) times the sub-step's CFL.
  - θ = 1 has a price. The interior L∞ error on #2468's σ = 0.1 oracle is 1.3 to 1.5 times larger, and a pair whose HJB
    half does not sub-step ran away faster on #1880's fixture (#2465). In its favour, HJB-SL's own error on a
    Cole–Hopf oracle is 7% to 39% smaller.
  - Stated at `FPSLSolver`'s `diffusion_theta`, pointed to from `HJBSemiLagrangianSolver`'s and `SLConfig`'s, and in
    `docs/user/CONVENTIONS.md` § 9's FP-SL note.
