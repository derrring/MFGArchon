- **`HJBGFDMSolver(monotonicity_scheme="joint_socp")` refuses an interior row its joint SOCP cannot place**
  (#1689). Such a row got the relaxed SOCP's weights, which carry slack on the M-matrix or the cone constraint, so
  it is not monotone, and it was counted as feasible. A node within `visibility_margin` of `obstacle_sdf`, which the
  visibility filter removes from its own neighbourhood, got no row and took bare Wendland-Taylor weights at solve
  time, silently. Construction now raises, naming the rows -- their count, and the first ten with coordinates and
  reasons -- and the way forward for each reason: for an infeasible SOCP, `socp_max_stencil_enlargements`, whose
  added neighbours are visibility-filtered only when `obstacle_sdf` is passed, or the new
  `socp_allow_relaxed_rows=True`; for a node in the margin, moving it out or reducing `visibility_margin`. The
  opt-in gives relaxed rows the relaxed SOCP's weights as before, reports them in one WARNING with their count and
  the largest `eps_M`, and does not count them as feasible. `PrecomputedJointSocpStencils` keeps them in
  `relaxed_rows` and the rows it could not place in `unplaced_rows`, and its `n_feasible` counts exact rows only.
