- **The network HJB solvers have their viscous term** (#1544). `NetworkHJBSolver` and
  `NetworkPolicyIterationHJBSolver` now solve `-u_t - D Lap_G(u) + H = 0`, where D = σ²/2 is read per solve from
  `volatility=` and `Lap_G = -L`, with L the problem's graph Laplacian (deg − A), the operator `FPNetworkSolver`
  applies as `D Lap_G(m)`. Before, a nonzero volatility was refused, because the HJB had no viscous term. None reads
  the problem's volatility, 0.0 on a `NetworkMFGProblem`, so a solve that passes none is unchanged. A scalar only, as
  FP-network takes; an array or callable volatility is refused.
  - Measured on a weighted 8-cycle against a manufactured solution (σ = 0.6, active control):
    - the ODE path matches to 1.45e-11;
    - policy iteration converges at first order in dt (1.21e-2, 6.05e-3, 3.03e-3 at Nt = 20, 40, 80).
  - **For a coupled solve, pass the same `volatility=σ` to `solve_fp_system` and to `solve_hjb_system`.**
    `FPNetworkSolver`'s `diffusion_coefficient=` is that solver's only, and the HJB does not read it. Its #1532
    warning now says so: it had recommended `diffusion_coefficient=`, which left the HJB at D = 0, the mismatched
    system #1544 reported.
