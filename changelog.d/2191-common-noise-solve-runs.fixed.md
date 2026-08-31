- **`CommonNoiseMFGSolver.solve()` can complete a conditional solve** (Issue #2191).

  It could not run at all, blocked twice over:

  `StochasticMFGProblem.create_conditional_problem` built an empty `MFGComponents` and then attached
  `hamiltonian_func` / `hamiltonian_dm_func` to it. Neither is a field of `MFGComponents`, so those
  lines set attributes nothing reads, and the constructor validates that a Hamiltonian or Lagrangian
  is present — so it raised before reaching them. The conditional Hamiltonian is now a real
  `HamiltonianBase` with the noise realisation bound into it, which makes the conditional problem an
  ordinary deterministic MFG to everything downstream. That is an adapter rather than a cast: the
  base class evaluates `(x, m, p, t)` of values, while the old component callables took
  `(x_idx, m_at_x, p_values, t_idx)` of grid indices with `p` arriving as a forward/backward dict.

  Behind that, the default `conditional_solver_factory` returned `prob.solve(verbose=False)` — a
  result — while its declared type is `Callable[[MFGProblem], MFGSolverProtocol]` and the caller
  invokes `.solve()` on what it returns. The default now satisfies the protocol it declares.

  Pinned by `tests/integration/test_common_noise_solve_runs_2191.py`. The load-bearing test makes
  the Hamiltonian depend on the noise and asserts the samples differ: the obvious fixture,
  `0.5 * p**2 + 0.1 * m`, has no `theta` in it, runs clean, and reports a Monte-Carlo spread of
  exactly 0.0 — so it would certify a `create_conditional_problem` that silently dropped the noise,
  which is precisely what the dead assignment did.
