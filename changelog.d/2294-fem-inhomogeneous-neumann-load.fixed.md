- **`HJBFEMSolver` assembles the inhomogeneous Neumann boundary load; `FPFEMSolver` refuses it**
  (Issue #2294). `du/dn = g` with `g != 0` owes the weak form `D * int_dOmega g phi_i` and nothing
  assembled it, so `g=0` and `g=5` gave bit-identical solutions while both solvers reported
  `honors_inhomogeneous_neumann = True`. For the HJB weak form, which integrates only
  `-D*Delta u` by parts, `NEUMANN(g)` is `ROBIN(alpha=0, beta=1, g)` and `assemble_robin_terms` now
  owns that load too. The FP weak form assembles `div(v m)` on the volume basis with no facet term,
  so its natural condition is the **total flux** `J.n`, not `dm/dn`: the same load would impose
  `J.n = -D*g`, a different condition. `FPFEMSolver` therefore declares
  `honors_inhomogeneous_neumann = False` and the Issue #1686 gate refuses such a problem before the
  solve instead of silently solving another one.
