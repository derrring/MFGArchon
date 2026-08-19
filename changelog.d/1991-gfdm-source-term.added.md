`HJBGFDMSolver.solve_hjb_system` accepts `source_term`, so a manufactured solution reaches GFDM
through **both** inner solvers. The capability gate keys on that parameter name and had been
rejecting GFDM for lacking a channel it already had under the name `running_cost`. Their signs are opposite
(`additive_source = -source_term`, since `h_eval` assembles `-u_t + H(+additive_source) - D*lap_u`
while the source contract subtracts), and the solver converted between them at the call site.
[SUPERSEDED by #1999, in this same release: the `running_cost` channel is gone, so only
`source_term` remains and there is nothing left to hold separately.] Measured with the 1D reduction of
the GFDM paper's manufactured pair: EOC 2.00/1.99 on the Newton path and 1.98/1.99 on the Howard
path, with the sign flipped staying flat at 1.42 on both — which is what the tests assert. The rate
is the SPATIAL order at fixed `nt`; the manufactured `a1(t)` is linear so backward Euler is exact in
time and no number here bears on the time discretisation. A source returning the wrong shape now
raises instead of being reshaped, since a 2D array in the wrong point order has the right size and
silently produces a different value function.
