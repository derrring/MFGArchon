- **`HJBGFDMSolver(monotonicity_scheme="joint_socp")` refuses an interior row its joint SOCP cannot place**
  (#1689). Such a row got the relaxed SOCP's weights, whose M-matrix slack `eps_M` is positive, so it is not
  monotone, and it was counted as feasible; a row with no neighbourhood was skipped silently. Construction now
  raises, naming the rows -- their count, and the first ten with coordinates and reasons -- and the two ways
  forward: `socp_max_stencil_enlargements`, whose added neighbours are visibility-filtered only when `obstacle_sdf`
  is passed, and the new `socp_allow_relaxed_rows=True`. The opt-in gives such rows the relaxed SOCP's weights as
  before, reports them in one WARNING with their count and the largest `eps_M`, and does not count them as
  feasible. `PrecomputedJointSocpStencils` keeps them in `relaxed_rows` and the rows it could not place in
  `unplaced_rows`, and its `n_feasible` counts exact rows only.
