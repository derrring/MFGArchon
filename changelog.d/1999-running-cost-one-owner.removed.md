`HJBGFDMSolver.solve_hjb_system` no longer accepts `running_cost`. Supplied alongside a Hamiltonian
that already held a potential, it double-counted silently (#2001), and that is what the removal
fixes.

Not that a second channel could *only* have carried the same quantity — it could carry two things
the Hamiltonian route does not deliver today, and both are open: an alpha-free `F(x, m)`, which
`SeparableHamiltonian` cannot express because its `coupling` takes only `m` (#2010); and anything at
all on the Howard path, whose closure keys on `SeparableHamiltonian` private attributes and drops a
general `HamiltonianBase` subclass bitwise (#2011). Neither is a reason to keep a double-counting
channel. `source_term` does reach Howard's closure — that is what #1991 added — so #2011 costs
nothing for `m`-free forcing; what it costs is model data, because `source_term`'s contract forbids
depending on `m`, and that is also why it cannot stand in for #2010.
The owner of an alpha-free `F(x, m)` is the Lagrangian, which is where the literature puts it. The package's only caller of that channel used it as an MMS source and has moved
to `source_term`, which #1991 added for exactly that. **Migrating callers must negate:
`source_term = -running_cost`.** `h_eval` assembles `-u_t + H(+additive_source) - D*lap_u` while
the package-wide source contract is `F(u) = (u-u_next)/dt + H - S = 0`, so the two slots carry
opposite signs. Passing the old callable unchanged solves a different problem and nothing raises. `h_eval.assemble_hjb_residual`'s parameter is
renamed `additive_source` to match what it now carries, and the orphaned `_normalize_running_cost`
machinery is deleted. `HJBHowardSolver`'s `running_cost=` is unchanged. It is public — the class is exported and
its `__init__` still takes the argument — but it is not a rival owner at that layer, because
Howard has no Hamiltonian of its own and GFDM feeds it the decomposition. So the count this
removal moves is one parameter on one solver, not every `running_cost=` in the package.
