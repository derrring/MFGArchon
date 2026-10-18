- **`HJBGFDMSolver(monotonicity_scheme="positive_qp_fan", inner_solver="howard")`: the positive-QP Laplacian and
  the guarded fan, with Howard's exact sectorwise policy** (#1689 PR 3).
  - A node's Laplacian row is the positive P2 row (`min ||L||^2`, `A^T L = e_lap`, off-centre `L_j >= 0`) over its
    21, 31, 41 or 61 nearest nodes. Its controls range over the sectors of a guarded fan, rays whose adjacent gaps
    lie between 25 and 120 degrees, or up to 175 within about 0.3h of a boundary. Every interior row of the
    policy-evaluation matrix is then an M-matrix row, for every policy.
  - Each Laplacian row is admitted by its certificate (each moment's residual and the off-centre signs, relative
    tolerance 1e-8, finite weights), never by a solver's status; off-centre weights within the tolerance below 0 are
    set to 0. A node no pool or fan places is refused by name.
  - `HJBHowardSolver(discretisation="guarded_fan", control_cost=...)` is the discretisation it runs on; its policy
    is the exact argmin over each node's sectors, the projection of `control_cost.optimal_control` onto the
    sector's cone.
  - Parameters: `positive_qp_fan=GuardedFanParameters(...)`, from `mfgarchon.alg.numerical.gfdm_components`, with
    defaults at the paper's values. A value outside the range the scheme supports is refused by name, with the
    reason.
  - Scope: two-dimensional, `SeparableHamiltonian` with `QuadraticControlCost`, and Dirichlet faces only. A
    boundary point of any other condition is refused by its classified type, at construction and at each solve: a
    reflecting face's BC row is a GFDM normal-gradient stencil with mixed signs, so it waits for the scheme's
    reflecting closure. Each step's BC rows are certified to have the M-matrix sign pattern where Howard installs
    them. Ghost nodes are refused, as with `joint_socp`.
  - Oracle: the decoupled LQ HJB in closed form converges at first order.
- The warning for a missing `monotonicity_scheme` recommends `positive_qp_fan` for 2-D problems with Dirichlet faces
  and a `SeparableHamiltonian` with a quadratic control cost, and `joint_socp` otherwise.
