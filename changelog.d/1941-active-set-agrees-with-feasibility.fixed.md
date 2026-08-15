`get_active_set` reported a point that *violates* a constraint as **inactive**. It tested
`|u - psi| < tol`, which is two-sided, while `is_feasible` tests one side of the inequality — so with
`lower = -1` and `u = [-2, -1, 0, 1, 2]` the field was infeasible and only the point exactly on the
bound was active, while the point violating by 1.0 was not. An active-set method runs on infeasible
iterates by construction, since that is the state before projection, and it was told to ignore
exactly the points that needed attention.

Both predicates now derive from the same inequality. `BilateralConstraint.get_active_set` carried a
third copy of the two-sided form and is fixed with them.
