`HJBFDMSolver(constraint=...)` now applies the constraint inside the backward sweep in 1-D, as it already did in n-D and as its docstring said (#2036).

**1-D used to clip the finished array.** The 1-D path projected every slice after the solve had returned. The sweep therefore never saw the obstacle, and the result was exactly `max(U_free, psi)`: a feasible array, but not a solution of the obstacle problem. Each step's Newton solution is now projected before the next, earlier step uses it, so the obstacle propagates backward through the sweep.

On a ceiling that binds on 10 nodes at t = 0, the per-step solution differs from the old clipped one by up to 3.5e-3.

**An infeasible terminal condition is now refused.** The terminal slice is the caller's data and is returned as given:

- before, n-D returned it unprojected and infeasible, while 1-D silently projected it;
- now a `U_terminal` outside the constraint set raises, with the size of the violation.

If the projected terminal condition is the one you mean, pass `constraint.project(U_terminal)`.

**What moves.**

- 1-D constrained solves change wherever the constraint binds before T.
- A call whose terminal condition violates the constraint now raises. Two tests in `test_hjb_with_obstacle.py` did so; they now use a problem whose obstacle binds inside the interval from feasible terminal data.
