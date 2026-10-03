A boundary condition set on the geometry after a solver is constructed is now checked before that solver solves it (#1699).

`_validate_bc_support` ran only in constructors, but four solvers read the geometry's BC again after construction:

- HJB-FDM, HJB-SL and FP-SL re-read it at every solve;
- HJB-GFDM's per-solve refresh adopted a newly set BC the same way.

So a BC set afterwards was solved without being checked, including a type the constructor refuses. A Neumann value was no exception: FP-SL drops one, and HJB-SL applies it only in part, which is why both constructors refuse it.

`get_boundary_conditions` now runs the check on every read, and raises `NotImplementedError` as the constructor would. HJB-FDM also reads it at the start of `solve_hjb_system`, so the n-D path refuses before the Newton residual, whose handler would retype the refusal (#2477).

**What moves.**

- A solve that used to return an answer for an unsupported BC now raises.
- Supported BCs, including a swap between two of them, solve exactly as before.
- The check costs about 2 µs per read. HJB-SL reads the BC at grid points within each time step. On a 2-D 41×41 grid with `Nt=10`, two fixtures measured 20,189 and 30,293 reads per solve, and a 5.6% and 6.5% longer solve.

**Not changed.** Five solvers solve the BC they captured at construction: WENO, FP-FDM, FP-FVM, FP-SL-Jacobian and particle FP. A geometry swap does not reach them, and assigning `solver.boundary_conditions` is still unchecked (#2475).
