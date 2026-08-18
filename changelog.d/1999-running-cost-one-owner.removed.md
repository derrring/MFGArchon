`HJBGFDMSolver.solve_hjb_system` no longer accepts `running_cost`. The alpha-independent part of
the Lagrangian — the potential `V(x,t)` and the coupling `f(m)` — is owned by the Hamiltonian and
already reaches the residual through it, so a second channel could only carry the same quantity a
second time: supplied alongside a Hamiltonian that already held a potential it double-counted
silently (#2001). The package's only caller of that channel used it as an MMS source and has moved
to `source_term`, which #1991 added for exactly that. `h_eval.assemble_hjb_residual`'s parameter is
renamed `additive_source` to match what it now carries, and the orphaned `_normalize_running_cost`
machinery is deleted. `HJBHowardSolver`'s `running_cost=` is unchanged. It is public — the class is exported and
its `__init__` still takes the argument — but it is not a rival owner at that layer, because
Howard has no Hamiltonian of its own and GFDM feeds it the decomposition. So the count this
removal moves is one parameter on one solver, not every `running_cost=` in the package.
