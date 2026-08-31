- **`CommonNoiseMFGSolver.solve()` can complete a conditional solve** (Issue #2191).

  It could not run at all, blocked twice over:

  `StochasticMFGProblem.create_conditional_problem` built an empty `MFGComponents` and then attached
  `hamiltonian_func` / `hamiltonian_dm_func` to it. Neither is a field of `MFGComponents`, so those
  lines set attributes nothing reads, and the constructor validates that a Hamiltonian or Lagrangian
  is present — so it raised before reaching them. The conditional Hamiltonian is now a real
  `HamiltonianBase` with the noise realisation bound into it, which makes the conditional problem an
  ordinary deterministic MFG to everything downstream ~~in every respect~~ — with the terminal
  condition being the exception that had to be fixed separately, below. That is an adapter rather than a cast: the
  base class evaluates `(x, m, p, t)` of values, while the old component callables took
  `(x_idx, m_at_x, p_values, t_idx)` of grid indices with `p` arriving as a forward/backward dict.

  Behind that, the default `conditional_solver_factory` returned `prob.solve(verbose=False)` — a
  result — while its declared type is `Callable[[MFGProblem], MFGSolverProtocol]` and the caller
  invokes `.solve()` on what it returns. The default now satisfies the protocol it declares.

  Pinned by `tests/integration/test_common_noise_solve_runs_2191.py`. The load-bearing test makes
  the Hamiltonian depend on the noise and asserts the samples differ: the obvious fixture,
  `0.5 * p**2 + 0.1 * m`, has no `theta` in it and runs clean, so it would certify a
  `create_conditional_problem` that silently dropped the noise, which is precisely what the dead
  assignment did. ~~and reports a Monte-Carlo spread of exactly 0.0~~ — only when `K` makes the
  sample mean exact; the assertion is now on the pairwise sample separation, which is exactly zero
  whenever the noise is dropped, at every `K`.

- **The conditional problem inherits the parent's terminal condition** (Issue #2191, found in
  review). `StochasticMFGProblem` normalised the terminal cost as `terminal_cost or g`, and
  `MFGProblem` defines neither, so the chain was unconditionally `None` and every conditional solve
  used `u_T ≡ 0` whatever the parent specified. Measured: `u_mean` was bit-identical for
  `u_terminal = 0.5x²` and for no terminal cost at all, against an `m_initial` control on the same
  surface that moves it by 1.988e-03. Because the solve then completed and reported convergence,
  the result was a confidently wrong number rather than a crash — and it only became reachable when
  `solve()` started working, so this fix ships with the one above rather than after it.

- **`parallel=True` no longer mis-pairs `u_samples` / `m_samples` with `noise_paths`** (Issue #2191,
  found in review). Results were collected with `as_completed` and appended, so they landed in
  completion order while `noise_paths` stayed in sampling order; the submission index was captured
  and never read. Measured: the recovered permutation was non-identity in 4 of 4 trials against a
  sequential control that was identity in 2 of 2. The aggregates (`u_mean`, `u_std`, the MC errors)
  are permutation-invariant and hid it, so only per-sample analysis was corrupted — and it
  re-permuted every run, so a fixed seed did not make it reproducible. `parallel=True` is the
  constructor default and the class docstring's own example.

- **The frozen-noise Hamiltonian adapter checks its batch size** instead of passing any non-scalar
  result through. A pointwise `0.5*p**2 + theta*m` broadcasts `(N,1)` against `(N,)` to `(N,N)` when
  the caller passes a batch; measured, `HamiltonianBase.dm` then returned `N²` values with no
  exception, and only a defensive `try/except` in the base class kept the answers right.
