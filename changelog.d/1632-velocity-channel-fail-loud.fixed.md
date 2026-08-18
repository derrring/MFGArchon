FP FDM: supplying a non-zero `velocity_field` to an advection scheme that cannot read it now
raises instead of silently solving at zero drift. Only `divergence_upwind` reads
`interface_velocity`; the other three ignored it *and* had their U channel already replaced by a
zero-U dispatcher, so the solve returned a finite, mass-conserving, converged-looking
pure-diffusion density. That was the reachable path for every non-separable Hamiltonian, since
the coupling layer routes those down the velocity channel precisely because `-c*grad(U)` cannot
represent their drift. A zero velocity is still accepted, since discarding it changes nothing.
Two integration tests were passing their drift through that dead channel and have been rerouted
to `potential_field`; the #1149 boundary-flux pin now actually pushes mass into the wall.
