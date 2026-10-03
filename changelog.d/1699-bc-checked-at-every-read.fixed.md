A boundary condition set on the geometry after a solver is constructed is now checked before that solver solves it (#1699).

`_validate_bc_support` ran only in constructors. HJB-FDM, HJB-SL and FP-SL read the geometry's BC again at every solve, so they solved a BC set afterwards without checking it. That included a type their constructors refuse, and a Neumann value the SL pair would drop. HJB-GFDM's per-solve refresh adopted such a BC the same way.

`get_boundary_conditions` now runs the check on every read, and raises `NotImplementedError` as the constructor would.

**What moves.** A solve that used to return an answer for an unsupported BC now raises. Supported BCs, including a swap between two of them, solve exactly as before.

**Not changed.** WENO, FP-FDM, FP-FVM, FP-SL-Jacobian and particle FP solve the BC they captured at construction. A geometry swap does not reach them, and assigning `solver.boundary_conditions` is still unchecked (#2475).
