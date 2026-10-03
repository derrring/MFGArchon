The finite-difference HJB Jacobian no longer clips the momentum at |p| = 1e6 (#1884).

Where |p| exceeded the limit, both ±ε probes clipped to the same value, so that row's Hamiltonian block came out zero: the Jacobian of a different operator, exactly where the gradient is steepest. Nothing reported it.

On a state with |p| ≈ 2e6, the interior rows' relative error against a difference of the residual was 1.00 with the clip. Without it the error is at most 4.9e-5, the round-off of the Jacobian's own ε.

This path is no longer the NumPy default. It still runs on non-NumPy backends and with `analytic_jacobian=False`, where Newton steps on steep states now move differently.

The unused helpers `_clip_p_values` and `_calculate_p_values` are removed.
