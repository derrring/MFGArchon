`HJBFDMSolver`'s 1-D Newton step now takes the batch path by default on the NumPy backend (#1884): the Hamiltonian is evaluated on every node in one call, and the Jacobian is the analytic (chain-rule) one. `analytic_jacobian` now defaults to `None`, meaning "batch on NumPy, per-point elsewhere". `True` and `False` keep their meaning.

The per-point path it replaces assembles the Jacobian by finite differences, O(Nx^2) per Newton step. #1607 kept the batch path opt-in because it converged on fewer problems; that no longer reproduces.

Measured at 5a16d60f across 27 coupled LQ configurations (Nx 31–81, coupling 0.02–2, sigma 0.05–0.4, no-flux and periodic, smooth and kinked terminal data):

- **19 configurations converged with both**, to the same answer within 1.2e-12 and in the same number of Picard iterations.
- **The other 8 converged with neither** (sigma 0.05 with coupling 0.5 or 2). They stop at different unconverged iterates and report `converged=False` either way.
- **Speed.** Solves are 6.4–11.6× faster.

**What moves.**

- Converged results move within the Newton tolerance: by at most 1.2e-12 on the configurations above, and by 1.3e-8 (relative 4e-10) on the towel-beach example.
- A solve that does not converge returns a different unconverged iterate.
- **A Hamiltonian that is not batch-safe is refused.** In batch, every node's value must equal that node evaluated alone. `0.5*p[0]**2` fails this: in batch it reads node 0's momentum at every node, and without the check it solved the wrong equation with no error (0.962 off on one fixture). A Hamiltonian whose batch call raises is refused the same way, with the reason. This includes the library's own `DualHamiltonian` from `legendre_transform()` (#2480). Write the Hamiltonian row-wise, as in `0.5*np.sum(p**2, axis=-1) + theta*m`, or pass `analytic_jacobian=False`.
- Couplings inside a `SeparableHamiltonian` are still evaluated point by point where they need it, and solve identically on both paths.
- `examples/applications/economics/towel_beach_demo.py` now writes its Hamiltonian row-wise.
- To restore the previous behaviour, pass `analytic_jacobian=False`.
