`FPParticleSolver` accepts `seed=`, so a stochastic solve can be repeated. Every draw it makes goes
through one owner, `self._rng` — a private `np.random.Generator` when `seed` is given, the global
`np.random` module when it is not.

`seed=None` keeping the global stream is plain backward compatibility, and that is the only claim
made for it: nine files here seed the global stream and then build this solver. The alternative was
measured rather than assumed — an entropy-seeded private default gives **1 failed / 65 passed**
across them, and that one failure (`test_issue_1412_fp_particle_sigma_override`) is **loud**. One
code path serves both, because `choice`, `uniform`, `normal` and `standard_normal` exist on the
module and on `Generator` with the same keywords.

**What this unblocks.** The #1822 capability matrix could not classify this solver in *either*
direction — three trials of one identical configuration returned `monotone = False, False, True`.
It was skipped by name in `STOCHASTIC_UNSEEDED`, which is now empty. With a seed, all four of its
declared boundary types converge under refinement, **20 of 20 seeds**, so none is listed as
unhonoured.

That verdict required fixing the refinement path as well as the seed. A particle method converges
as `N → ∞` *and* `h → 0`; the matrix refined only the grid, leaving a Monte Carlo floor set by `N`,
so at the fine end its convergence oracle was comparing noise — the PERIODIC row came out monotone
in only 6 of 12 seeds, a coin flip written into a strict ratchet. The fixtures now scale
`num_particles ~ Nx²` alongside the grid, the standard pairing since the `O(N^-1/2)` sampling error
has to track `O(h)`.

The solver still fails the **absolute** seam tolerance at a single coarse grid, and that stays an
`xfail`: at `Nx=21` the seam has a floor no particle count removes — 5.22e-01 (N=2e3), 3.95e-01
(2e4), 3.32e-01 (2e5), 3.05e-01 (2e6). Driving the KDE bandwidth to zero isolates it at ≈2.7e-01,
where it is both bandwidth- and `N`-independent, so it is a boundary-treatment bias rather than
sampling error. About half of it is the Issue #709 boundary smoothing, applied unconditionally:
with `kde_boundary_smoothing=False` the seam drops 3.29e-01 → 1.48e-01 at N=2e5. That is a
fixed-grid property of Monte Carlo + KDE, not a failure to honour the boundary condition.
