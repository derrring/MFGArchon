FP FDM: a `velocity_field` that cannot be honoured now raises instead of silently solving at zero
drift. Two hazards, kept separate. **Displacement** — the velocity branch wins over both
`U_solution_for_drift` and a callable `drift_field`, so supplying a velocity alongside either
silently discards it; this bites even on `divergence_upwind`, where a zero velocity is consumed as
zero advection while the callable drift is thrown away. **Consumption** — an unconsumed velocity is
dropped outright, harmless only when it is all zeros. `_velocity_is_consumed` now also requires
scalar diffusion: `solve_timestep_tensor_explicit` takes no `interface_velocity`, so on the
tensor path no scheme reads the velocity, including the only one on the accept-list. Two
integration tests were passing their drift through the dead channel and have been rerouted to
`potential_field`, with a bump at each wall and a curved potential so both wall stencils are
exercised — reverting either wall to the pre-#1149 one-sided face velocity now fails that test,
where a linear potential left both stencils algebraically identical and a single-wall bump left
the right wall carrying 5e-15 of density.
