FP FDM: supplying more than one drift input to `solve_fp_nd_full_system` now raises. The dispatch
is a precedence order and every level silently discards the ones beneath it — `velocity_field`
wins over both siblings, and inside the second arm a callable `drift_field` wins over
`U_solution_for_drift` — so the solve ran on one input and reported nothing about the others.
Separately, a non-zero velocity that no scheme will read still raises: `_velocity_is_consumed` now
also requires scalar diffusion, because `solve_timestep_tensor_explicit` takes no
`interface_velocity` and on that path even `divergence_upwind`, the default and sole member of the
accept-list, reads nothing. Two integration tests were passing their drift through the dead channel
and have been rerouted to `potential_field`, with a bump at each wall and a curved potential so
both wall stencils are exercised: reverting either wall to the pre-#1149 one-sided face velocity
now fails that test, where a linear potential left the two stencils algebraically identical and a
single-wall bump left the right wall carrying 5e-15 of density. Scheme-name resolution and its
error string now have one owner.
