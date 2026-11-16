- **A value or callable that is NaN or infinite where the solvers will read it is refused where it enters, naming it**
  (#2974). `MFGProblem` reads each once while it is built, and only at states that every `NumericalScheme` that solves
  the problem hands that callable (`docs/user/CONVENTIONS.md`, "What construction refuses as non-finite"):
  - **the arrays and scalars whole**: `T`, `m_initial`, `u_terminal`, the volatility and `state_penalty_scale`;
  - **each callable on the time grid's nodes**: the interior nodes for every callable, with t_0 for the potential, the
    Hamiltonian's other time-dependent sources, a callable volatility and a boundary callable, and T alone for the
    terminal coupling. `source_term_hjb`, `source_term_fp` and `coupling_cost` are read at the interior nodes alone;
  - **at the probe nodes off the boundary**, with `m_initial`'s and `u_terminal`'s values there.

  A test records what `problem.solve(scheme=...)` hands each callable under every member of the enum, and holds these
  reads inside it.
  - **Refused at construction now, where the solve used to return a finite wrong U with `converged` False, a
    non-finite U, or an error naming something else** ("Check CFL condition"), when the value reaches those states:
    `state_penalty`, `state_penalty_scale`, `source_term_hjb`, `source_term_fp`, `nonlocal_operator`, `coupling_cost`
    and `terminal_coupling`, a scalar volatility or `diffusion`, `T`, a `CongestionHamiltonian`'s `congestion_factor`,
    a Hamiltonian's `coupling` non-finite at `m_initial`'s values, a `NetworkHamiltonian`'s `node_potential_func`, a
    boundary segment's callable value, and a `NodeBC`'s callable value.
  - **Refused where its object is built:** a radial kernel's NaN parameter, and its infinite amplitude or exponent (an
    infinite length scale or softening stays accepted: it solves as the constant or zero kernel it tends to);
    `BoundedControlCost(max_control=nan)` (`max_control=inf` stays accepted: it solves as unbounded);
    `QuadraticMFGHamiltonian(coupling_coefficient=nan)`; and a `BCSegment` whose constant is NaN or infinite in a
    field its type reads (`FIELDS_READ`: the value of a Dirichlet, Neumann, Robin or flux segment, and a Robin
    segment's `alpha` and `beta`), which the solve used to refuse where it read it.
  - **Refused at the solver's door:** a NaN or infinite `volatility=` handed to `solve_hjb_system` or `solve_fp_system`,
    scalar or array, beside the negative one (#2618).
  - **Named by the source now, where the message said "Hamiltonian returned NaN or Inf at sample point" or named the
    array's validator:** a Hamiltonian's or a `SeparableLagrangian`'s `potential` and `coupling`, `m_initial`,
    `u_terminal` and a volatility array.
  - **Built now, where it was refused:** a Hamiltonian whose `coupling`, or whose own `__call__`, is non-finite only at
    m = 1, with an `m_initial` that is 1 at no node. Both problems solve to the same U and M as their finite twins,
    measured with `m_initial` 0.75 (1 + x²).
  - **`validate_hamiltonian` reads its one state at a new place:** m is `m_initial`'s value at its node, which it read
    as 1. x is the centre probe node off the boundary, which it read as the grid's middle index. On a grid with an even
    node count on an axis that index can lie on the boundary: on 6 × 6 it is (0.6, 0). An own `__call__` non-finite only
    there is no longer refused at construction, and the solve reaches it (#2975).
  - **Direct calls:** `validate_m_initial` and `validate_u_terminal`, which `validate_components` calls, no longer report
    a non-finite value; the problem's construction decides it. `validate_hamiltonian` and `validate_custom_functions`
    take `m_initial=`, and without it read m = 1 as before.
  - **For subclasses:** `HamiltonianBase` and `LagrangianBase` carry `HELD_SOURCES`, `CHECKED_SCALARS` and `NOT_HELD`.
    A source a class lists in `HELD_SOURCES` is read by construction; the library's classes list every value their
    constructor takes, and a test holds them to it.

  **Not seen at construction:**
  - a source finite at these reads and non-finite elsewhere: at a boundary node, which the solve reads under no-flux;
    at another time, m or v; or at a state the solve reaches;
  - a `source_term_fp` that takes v, and a source non-finite only at t_0 (`source_term_hjb`, `coupling_cost`) or only
    at T (`source_term_fp`), which some scheme that solves the problem does not read;
  - a Hamiltonian's own `dp` and `optimal_control`;
  - at Nt = 1, `source_term_hjb`, `source_term_fp` and `coupling_cost`;
  - on a grid with an axis of 2 nodes, which has no probe node off the boundary, the callables read there: a callable
    volatility, the held sources, the two source terms and `state_penalty`;
  - what an expert-mode solver handed to `problem.solve` reads and no scheme of the enum does;
  - on a network, a source that takes p or m (`hamiltonian_func`, `node_interaction_func`), which the solvers are
    handed by their caller;
  - on an implicit domain, the callables, since such a domain draws its points afresh at each read of its grid (#2778).

  What the solve reaches is the point-of-use checks' (#2975, #2976).
