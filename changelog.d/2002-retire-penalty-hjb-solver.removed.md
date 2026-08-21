**BREAKING.** `PenaltyHJBSolver` is retired and its constructor now raises `NotImplementedError`
(#2002). It was written to add the variational inequality `v >= Psi(x)` to any HJB solver by
injecting a penalty into that solver's `source_term`, but a penalty for that constraint is
`max(0, Psi - v)` and `source_term` is `(t, x) -> array` — there is nowhere for the value function
to enter. What it applied was `penalty_parameter * max(0, Psi(x))`, byte-identical at a node
satisfying the constraint and one violating it, and no value of `penalty_parameter` changes that.
The name still resolves and the error explains the situation and names `HJBFDMSolver(constraint=...)`
(#591) as the reachable alternative, with its own limits in #2036. Retiring it removes a capability
that was never there: it was the only mechanism claiming to give obstacle support to solvers other
than `HJBFDMSolver`, and #2046 tracks doing that properly by threading the constraint through the
shared timestep solve. Two further copies of the withdrawn "proper handling is the PenaltyHJBSolver
wrapper (#924)" claim, in `newton_mfg_solver` and `graph_mfg_solver`, are struck as well.
