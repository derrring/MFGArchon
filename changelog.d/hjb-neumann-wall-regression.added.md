A regression pin that the HJB solvers converge to the Neumann wall, via a symmetry oracle.

`d_n u = 0` at a reflecting wall is a *property* of the exact solution, so it needs no source term
and no exact-solution transcription. `u_T(x) = cos(2πx)` on `[0,1]` is even about both walls; under
zero coupling the HJB flow preserves evenness (`u_xx` does, and `|u_x|²` is the square of an odd
function), so `d_n u ≡ 0` exactly and any non-zero one-sided difference at the wall is
discretization error. The control is a constant terminal condition, which makes the same metric zero
to machine precision — measured `4.4e-15` or exact.

Measured at `Nx = 21/41/81`, `sigma=0.5`, `T=0.5`, `Nt=20`: `HJBFDMSolver` 1.29e-01 → 6.23e-02 →
3.05e-02 (orders 1.05, 1.03); `HJBWENOSolver` 4.23e-02 → 2.11e-02 → 1.06e-02 (1.00, 0.99);
`HJBSemiLagrangianSolver` 8.65e-02 → 4.11e-02 → 2.68e-02 (1.07, 0.62, net 0.85).

SL is asserted on monotone decrease plus a net order, not a per-level band: its irregularity is not
metric noise but adaptive substepping (33 / 112 / 89 substeps at different resolutions), so the
effective time discretization changes between levels. A per-level band there would be a threshold
fitted to one run.

Sensitivity measured rather than assumed — injecting a wall-gradient perturbation decaying as
`nx**-p`: `p=0.0` caught, `p=0.5` caught, `p=0.8` (the band's own edge) passes, `p=1.0` passes.

**This is a regression pin, not a bug hunt.** A 2026-08-17 measurement recorded two of these solvers
violating the condition at `O(1)` — ~1.0 still at `Nx=161`. That is no longer reproducible. The
configurations are not identical so no causation is claimed, but the earlier record's own open
question was whether `HJBFDMSolver`'s then-exact zero came from `_apply_neumann_enforcement`
overwriting the wall row, which #1902 deleted — and FDM now shows a converging `6.2e-02` rather than
an exact zero, consistent with the scheme converging instead of the value being painted on.

1D deliberately: `d_n u = 0` is scalar per wall with no tangential component, so it is fully
expressed in 1D per `AGENTS.md`'s dimension rule, and a 2D version would separate no extra mutant.
