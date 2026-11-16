- **A value or callable that is NaN or infinite where the solvers will read it is refused where it enters, naming it**
  (#2974). `MFGProblem` reads each once while it is built, at the state the solve is certain to read: `T`,
  `m_initial`, `u_terminal`, the volatility and `state_penalty_scale` whole, and each callable at the probe nodes and at
  0, T/2 and T, with `m_initial`'s and `u_terminal`'s values there (`docs/user/CONVENTIONS.md`, "What construction
  refuses as non-finite").
  - **Refused at construction now, where the solve used to return a finite wrong U with `converged` False, a
    non-finite U, or an error naming something else** ("Check CFL condition"): a non-finite `state_penalty`,
    `state_penalty_scale`, `source_term_hjb`, `source_term_fp`, `nonlocal_operator`, `coupling_cost` and
    `terminal_coupling` (a `PotentialCost` with V NaN or inf), a scalar volatility or `diffusion`, `T`, a
    `CongestionHamiltonian`'s `congestion_factor`, a Hamiltonian's `coupling` that is non-finite at `m_initial`'s values,
    a `NetworkHamiltonian`'s `node_potential_func`, `node_interaction_func` and `hamiltonian_func`, and a boundary
    segment's callable value.
  - **Refused where its object is built:** a radial kernel's parameters, `BoundedControlCost(max_control=nan)`
    (`max_control=inf` stays accepted: it solves as unbounded), `QuadraticMFGHamiltonian(coupling_coefficient=nan)`, and
    a `BCSegment` whose constant `value`, `alpha` or `beta` is NaN or infinite, which the solve used to refuse.
  - **Refused at the solver's door:** a NaN or infinite `volatility=` handed to `solve_hjb_system` or `solve_fp_system`,
    scalar or array, beside the negative one (#2618).
  - **Named by the source now, where the message said "Hamiltonian returned NaN or Inf at sample point" or named the
    array's validator:** a Hamiltonian's or a `SeparableLagrangian`'s `potential` and `coupling`, `m_initial`,
    `u_terminal` and a volatility array.
  - **Built now, where it was refused:** a Hamiltonian whose `coupling`, or whose own `__call__`, is non-finite only at
    m = 1, with an `m_initial` that is 1 at no node. `validate_hamiltonian` read `__call__` at m = 1, and now reads it
    at `m_initial`'s value at its node. Measured with `m_initial` 0.75 (1 + x²): both problems solve to the same U and M
    as their finite twins, and converge.
  - **Direct calls:** `validate_m_initial` and `validate_u_terminal` no longer report a non-finite value, which the
    problem's construction decides. `validate_hamiltonian` and `validate_custom_functions` take `m_initial=`, and
    without it read m = 1 as before.
  - **For subclasses:** `HamiltonianBase` and `LagrangianBase` carry `HELD_SOURCES`, `CHECKED_SCALARS` and `NOT_HELD`.
    A source a class lists in `HELD_SOURCES` is read by construction; the library's classes list every value their
    constructor takes, and a test holds them to it.

  Not seen at construction: a source finite at these reads and non-finite elsewhere, a Hamiltonian's own `dp` and
  `optimal_control`, and on an implicit domain the callables, since such a domain draws its points afresh at each read
  of its grid (#2778). What the solve reaches is the point-of-use checks' (#2975, #2976).
