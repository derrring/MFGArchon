- **The FP semi-Lagrangian solver states the norm its density converges in, and no longer states orders it does not
  have** (closes #2468; the scheme question is #2645).
  - `FPSLSolver`, at σ = 0, for a given drift:
    - the density converges in W₁, which is what the scheme's theory gives (Carlini–Silva, arXiv:1708.02042,
      Theorem 4.1: on ℝ^d without a boundary, under its (H)-(Lip) and dx² = o(dt), every limit point solves the
      equation, and at σ = 0 with a Lipschitz drift that solution is unique);
    - it does not converge pointwise where a component of the drift changes sign along its own axis. In 1-D the
      linear deposit gives a compressive stagnation point on a node twice the equation's rate, and the peak of a
      compressive drift stays too high under refinement: on #2468's fixture 4.16 to 4.20 against an exact 3.02 over
      nx = 51 to 801;
    - where σ = 0 drives mass into a wall, the exact law has an atom there and L¹ does not converge.
  - With σ > 0, away from walls, the density converges pointwise. Where mass reaches a no-flux wall, at the default
    `diffusion_theta`, it need not (#1855).
  - Orders that were stated as the density's accuracy are removed:
    - the splatting methods' "O(dx)", "O(dx³)" and "O(dx⁵)";
    - `NumericalScheme.SL_LINEAR`'s "O(h^2) with Δt = O(h^{3/2})" and `SL_CUBIC`'s "O(h^4)", which now state the FP
      half's convergence for a given drift and leave the pair's unstated;
    - the SL pair factory's comment "maintains O(h^2) convergence" (`_create_sl_pair`, behind `create_paired_solvers`);
    - the migration guide's "1st order" and "3rd order (HJB)" for the SL pairs;
    - `splat_nd`'s advice that linear splatting "provides adequate accuracy for most MFG applications";
    - `HJBSemiLagrangianSolver`'s stochastic step's "O(h) convergence on the 1D Boltzmann-Gibbs equilibrium against the
      Carlini-Silva theoretical O(h^2)", and the FP source term's "first order in time, matching the scheme's own
      order".

    The splatting methods are described by the interpolation each transposes.
  - `docs/user/CONVENTIONS.md` § 9 gains the rule: a density converges in a named norm, and a peak value or a plot of
    an FP-SL density is a pointwise reading.
