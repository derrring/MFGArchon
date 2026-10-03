`HJBFDMSolver`'s 1-D inner Newton step now uses the analytic (chain-rule) HJB Jacobian by default on the NumPy backend (#1884). `analytic_jacobian` now defaults to `None`, meaning "analytic on NumPy, finite differences elsewhere". `True` and `False` keep their meaning.

The per-point finite-difference Jacobian it replaces costs O(Nx^2) per Newton step. #1607 kept the analytic one opt-in because it converged on fewer problems; that no longer reproduces.

Measured across 27 coupled LQ configurations (Nx 31–81, coupling 0.02–2, sigma 0.05–0.4, no-flux and periodic, smooth and kinked terminal data):

- **19 configurations converged with both**, to the same answer within 1.2e-12 and in the same number of Picard iterations.
- **The other 8 converged with neither** (sigma 0.05 with coupling 0.5 or 2). They stop at different unconverged iterates and report `converged=False` either way.
- **Speed.** Solves are 6.4–11.6× faster.

**What moves.**

- Converged results move only at round-off.
- A solve that does not converge returns a different unconverged iterate.
- **1-D HJB-FDM now evaluates the Hamiltonian in batch on NumPy, as n-D already did.** A Hamiltonian or coupling written for scalars only, such as `0.5*p**2 + theta*m`, now raises instead of solving. Reduce over the control axis, as in `0.5*np.sum(p**2, axis=-1) + theta*m`.
- To restore the previous behaviour, pass `analytic_jacobian=False`.
