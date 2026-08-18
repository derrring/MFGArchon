FP FDM: supplying a `velocity_field` that an advection scheme cannot read now raises instead of
silently solving at zero drift. Only `divergence_upwind` reads `interface_velocity`; on the other
three the velocity was ignored *and* the U channel had already been replaced by a zero-U
dispatcher, so the solve returned a converged-looking pure-diffusion density. Reachable by an
explicit velocity — `FPFDMSolver.solve_fp_system(m0, drift_field=<ndarray>)` on a non-consuming
scheme, or `FixedPointIterator(..., drift_field=<ndarray>)`, which is forwarded verbatim — and not
on the auto-routed coupling path, where `fp_drift_coefficient` raises first (#1542). An all-zero
velocity with no U supplied is still accepted, since it displaces nothing; supplied alongside a
real U it raises, because the zero-U dispatcher would discard that U. Two integration tests were
passing their drift through the dead channel and have been rerouted to `potential_field`, with a
non-linear potential so the #1149 boundary-flux pin discriminates: reverting both walls to the
pre-#1149 one-sided face velocity now fails that test, where under a linear potential the two
stencils are algebraically identical and it stayed green.
