`HJBGFDMSolver.solve_hjb_system` accepts `source_term`, so a manufactured solution reaches GFDM
through **both** inner solvers. The capability gate keys on that parameter name and had been
rejecting GFDM for lacking a channel it already had under the name `running_cost`. The two are
deliberately not unified: their signs are opposite (`running_cost = -source_term`, since `h_eval`
assembles `-u_t + H(+running_cost) - D*lap_u` while the source contract subtracts), so the solver
holds them as separate attributes and adds them at the call site. Measured with the 1D reduction of
the GFDM paper's manufactured pair: EOC 2.00/1.99 on the Newton path and 1.98/1.99 on the Howard
path, with the sign flipped staying flat at 1.42 on both — which is what the tests assert. The rate
is the SPATIAL order at fixed `nt`; the manufactured `a1(t)` is linear so backward Euler is exact in
time and no number here bears on the time discretisation. A source returning the wrong shape now
raises instead of being reshaped, since a 2D array in the wrong point order has the right size and
silently produces a different value function.
