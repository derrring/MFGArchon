- **The SL pair's `diffusion_theta` documents what it trades at a no-flux wall** (#1855). The default does not change:
  per the user's answer of 2026-10-14, the trade-off is the caller's to configure.
  - At a no-flux wall, the split step's steady state depends on (1 − θ)·dt. The wall node's bias is the semi-discrete
    steady state's, which is first order in h, plus about K·(1 − θ)·|α|·dt/h, with K = 0.74 to 0.96 over nx = 41 to 321
    on #1855's fixture. So along dt ∝ h it does not refine away at θ = 0.5, and at θ = 1 it is absent.
  - θ = 1 has a price. The interior L∞ error on #2468's σ = 0.1 oracle is 1.3 to 1.5 times larger, and a pair whose HJB
    half does not sub-step runs away faster (#2465). In its favour, HJB-SL's own error on a Cole–Hopf oracle is 7% to
    39% smaller.
  - Stated at `FPSLSolver`'s `diffusion_theta`, pointed to from `HJBSemiLagrangianSolver`'s and `SLConfig`'s, and in
    `docs/user/CONVENTIONS.md` § 9's FP-SL note.
