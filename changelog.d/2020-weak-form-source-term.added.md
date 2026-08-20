`source_term` now reaches the weak-form solver family, and a class-definition-time gate stops a
future solver from swallowing it.

Six solvers accepted `source_term` through `**kwargs` and discarded it — no exception, no warning,
bitwise-identical output. They reduce to two definers, so two changes cover all six:
`WeakFormFPSolver.solve_fp_system` (used by `FPFEMSolver`, `MeshlessGalerkinFPSolver`) and
`WeakFormHJBSolver.solve_hjb_system` (used by `HJBFEMSolver`, and by
`MeshlessGalerkinHJBSolver`, which forwards to it).

The HJB side needed no new machinery: `_solve_timestep_newton` already had a `rhs_coupling`
parameter that its residual **subtracts**, and the documented contract is
`F(u) = (u - u_next)/dt + H - S`, so that is the source slot.

**Verified by order, not by liveness.** "Passing a source changes the answer" is a liveness check.
Measured against manufactured solutions on `[0, 1]` with no-flux walls, with the spatial and
temporal studies deliberately separated so neither caps the other:

| study | HJB picard | HJB newton | FP |
|---|---|---|---|
| space, P1, steady `u*` | 1.883 / 1.945 / 1.974 | 1.818 / 1.805 / 1.895 | 2.021 / 2.014 / 2.008 |
| time, backward Euler | 0.999 / 1.000 / 1.000 | 1.039 / 1.018 / 1.009 | 0.948 / 0.973 / 0.987 |

$O(h^2)$ in space for P1 and $O(\Delta t)$ in time for backward Euler, on both HJB branches.

The test file also pins the trap that produced those numbers on the second attempt: the first
temporal fixture was linear in `t`, and backward Euler's truncation error is $(\Delta t/2)u_{tt}$,
identically zero there. It reported order 0.02 from a correct implementation, because it was
measuring the spatial floor. `test_the_time_fixture_is_not_degenerate` asserts $u_{tt} \neq 0$.

**The gate.** An abstract method's signature is a declaration and Python does not enforce it — a
subclass may override with `(*args, **kwargs)` and nothing checks, which is the structural reason
this could exist unnoticed. `BaseHJBSolver` and `BaseFPSolver` now validate overrides at class
definition: an override that accepts `**kwargs` must NAME `source_term` and `volatility_field`.
Scoped to those two because they have the incident history (#1424, #2020; #1316, #1783); a blanket
"name every declared parameter" rule is not satisfiable, since the base declares
`m_initial_condition` while implementations use `m_initial` or `M_initial` — that is a rename, not a
gate. Solvers without `**kwargs` are untouched: an unnamed parameter there already raises
`TypeError`, which is loud.

A solver that cannot support a parameter still names it and raises inside. That is what
`HJBWENOSolver` does for multi-D `source_term`, and it is the honest shape: refusal is a behaviour,
an absent signature is not.
