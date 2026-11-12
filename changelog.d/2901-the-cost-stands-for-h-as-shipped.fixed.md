- **The control cost stands for the Hamiltonian only as the library ships both,** for the readers of the cost or of a Lagrangian built on it (#2901, stage 1b; user ruling, 2026-11-11). One owner decides it, `_why_not_as_it_ships` (#2914), where each reader decided by a test of its own.
  - **Refused:** a `SeparableHamiltonian` subclass and a user's cost (a subclass of a shipped one, or a shipped cost whose instance holds its own method, `effective_domain` included). The readers are HJB-SL's DPP path, on which a non-smooth separable Hamiltonian is solved, HJB-GFDM's `inner_solver='howard'` and `monotonicity_scheme='positive_qp_fan'`, and `HJBHowardSolver(discretisation='guarded_fan')`. Each refusal says what was run and on what.
  - **Before:**
    - a subclass whose `__call__` carried another term was solved without it on the DPP path, 1.173 from HJB-FDM's U, where the shipped Hamiltonian is 0.073 from it;
    - a cost redefining only its Lagrangian put Howard 0.325 off Newton, where the shipped cost is 2e-08 off.
  - **`MFGComponents` derives a Lagrangian from the cost only there,** also through a `regularize()` wrapper, so a user's cost the wrapper holds derives none. `MFGProblem.lagrangian_class` is now `None` for a Hamiltonian-defined problem whose Hamiltonian or cost is not as shipped, where it was a `SeparableLagrangian`.
  - **The FP route** asks the Hamiltonian, not its cost, whether it is smooth.
  - **HJB-SL's `diffusion_method='canonical_cs'`** reads the cost's `lambda_` only for a Hamiltonian and cost as the library ships them. It refuses a `SeparableHamiltonian` subclass, or a `QuadraticControlCost` subclass, even one that changes nothing, as FP-FDM's −c∇U route does since #2915; before, a subclass that redefined H was solved with the cost's answer. `BaseHJBSolver._control_cost_lambda` reads the same owner.
  - **HJB-GFDM's DMP warning** reads its drift from the Hamiltonian's `dp`, at the step's density and time, refused unless it has p's shape.
  - **Removed:**
    - `_control_cost_lambda`'s fallback to `problem.lambda_`, for a problem with no Hamiltonian or a Hamiltonian with no control cost;
    - HJB-SL's no-Hamiltonian branch of its characteristic velocity.

    A problem always carries a Hamiltonian (#670), and nothing but a stand-in in a test reached either.
  - **Not yet met (#2901, stage 2):** a `SeparableLagrangian` subclass given alone is still read through its own `__call__` on HJB-SL's DPP path and through its cost by the Hamiltonian derived from it.
