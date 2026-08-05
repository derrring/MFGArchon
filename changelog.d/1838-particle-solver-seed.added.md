`FPParticleSolver` accepts `seed=`, so a stochastic solve can be repeated. Every draw it makes now
goes through one owner, `self._rng`, which is either a private `np.random.Generator` (when `seed` is
given) or the global `np.random` module (when it is not).

`seed=None` is deliberately the global stream and not an entropy-seeded `Generator`: twelve files in
this repository call `np.random.seed(...)` and then build this solver to compare two runs. A private
default would silently stop honouring those seeds, and those comparisons would keep passing while
comparing nothing. One code path serves both, because `choice`, `uniform`, `normal` and
`standard_normal` exist on the module and on `Generator` with the same keywords.

**What this unblocks.** The #1822 capability matrix could not classify this solver in *either*
direction — three trials of one identical configuration returned `monotone = False, False, True`, so
`xfail` asserted a failure it did not reliably have and `pass` asserted the opposite. It was skipped
by name in `STOCHASTIC_UNSEEDED`, which is now empty. Measured with a seed, its four declared
boundary types split three to one:

| BC | residual at Nx=21/41/81 | verdict |
|:--|:--|:--|
| `NO_FLUX`, `NEUMANN` | 5.40e-02, 3.24e-02, 1.54e-02 | converges — honoured |
| `DIRICHLET` | 4.67e-01, 1.05e-04, 7.9e-169 | converges — honoured |
| `PERIODIC` | 4.95e-01, 2.66e-01, 2.79e-01 | **rises at 81 — not honoured** |

The periodic seam is a bias, not Monte Carlo noise: holding the grid at Nx=21 and raising the
particle count, it converges to a **nonzero** limit while the seed-to-seed spread collapses —
5.4e-01 (N=2e3), 3.9e-01 (N=2e4), 3.33e-01 (N=2e5), spread 4e-02 → 1.5e-03. Sampling error would
vanish. The mechanism is visible in the output: `f[0] - f[1]` and `f[-1] - f[-2]` are **exactly
zero**, so the KDE duplicates its edge bins. That is one broken column, not a broken solver, and it
is now an honest `xfail` with a repeatable number instead of a skip.
