`HJBGFDMSolver.solve_hjb_system` accepts `source_term`, so a manufactured solution reaches GFDM.
The capability gate keys on that parameter name, so it had been rejecting GFDM with "Use an FDM HJB
solver" for lacking a channel GFDM already had under the name `running_cost` — the gate was
measuring a proxy for the capability it was asked about. The two are deliberately NOT unified:
`running_cost` is model data and may depend on `m` (Howard documents it as the
non-alpha-dependent part of the Lagrangian — potential, congestion), while a source is artificial
forcing that may not. They share one arithmetic slot with opposite signs, `running_cost =
-source_term`, so the solver holds them as separate attributes and adds them only at the call site.
Measured with the 1D reduction of the GFDM paper's manufactured pair: EOC 2.00/1.99, and the same
study with the sign flipped stays flat at 1.42, which is what the accompanying test asserts.
