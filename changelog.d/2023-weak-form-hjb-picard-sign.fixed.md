The weak-form HJB solver's Picard branch — **the default** — applied the Hamiltonian with the wrong
sign, so `HJBFEMSolver` and `MeshlessGalerkinHJBSolver` solved `-u_t - H - D·Lap(u) = 0` on the path
a caller gets without asking.

The canonical equation (`mfg_problem.py:197`) is `-u_t + H - (sigma²/2)·Lap(u) = S`, so backward
Euler gives `(M/dt + D·K)·U[n] = (M/dt)·U[n+1] - M·H`. The line read `+=`.

Measured on `H = c` constant with `u_T = 0` and no-flux, where the solution is spatially constant and
`u(0) = -c·T` is exact:

| c | analytic | Picard before | Picard after | Newton | FDM |
|---|---|---|---|---|---|
| 1.0 | −0.200000 | **+0.200000** | −0.200000 | −0.200000 | −0.200000 |
| 2.0 | −0.400000 | **+0.400000** | −0.400000 | −0.400000 | −0.400000 |
| −3.0 | +0.600000 | **−0.600000** | +0.600000 | +0.600000 | +0.600000 |

Exactly `−1×` at every value. Independently confirmed against an external analytic oracle: with the
control cost switched off the PDE is linear with a closed-form solution, and the corrected tree
converges to it at second order (2.844e-03 → 1.383e-03 → 6.817e-04 → 3.384e-04, ratios 2.06 / 2.03 /
2.02) while the old one plateaus at 3.30e-01 with ratios 0.996 / 0.998 / 0.999.

**Provenance.** The line entered at `d9f66701` (#773), the file's first version — which stated the
correct algebra in a comment three lines above the code, `M/dt·u^n = M/dt·u^{n+1} - D·K·u^n - H_rhs`,
and then "rearranged" it to `+ source` on the very next line. `675e0049` (#1131) only relocated it.
The file shipped with its own derivation contradicting its code, on adjacent lines.

**Why five months of green.** Nothing ran the two branches on the same problem and compared. The one
test that checks the weak-form family against an independent discretization
(`test_meshless_stabilization.py`) passed `use_newton=True`, so the suite's only cross-implementation
oracle was pointed at the branch that is not the default. It is now parametrized over both.

Three tests changed, none by weakening a bound:

- `test_fem_robin_bc.py::test_hjb_linear_loop_fixed_point` built its reference by `spsolve`-ing the
  solver's own matrices against the same source the Hamiltonian carries (measured
  `max|H(x,m,p=0) − _source| = 0.0`) — it was the buggy right-hand side written down twice. The
  Hamiltonian potential is negated; the manufactured solution, the Robin data and the oracle line are
  untouched, and `max|U[0] − u_steady|` is now 1.010e-14 against the unchanged 1e-9 bound.
- The two coupled-FEM mass tests specified `coupling = lambda m: m`, which under this repo's
  reward-signed convention (`mfg_problem.py:191-195`) is crowd-**seeking**: an aggregating,
  anti-monotone MFG whose fixed point diverges. They passed only because the negated assembly turned
  an aggregating specification into a dispersing computation. Both now specify congestion
  (`-m`) and conserve mass to 2.2e-16. Verified pre-existing: their fixture through **main's own
  Newton branch, no fix applied**, diverges to a mass drift of 7.36e+26.

Also corrected: a recorded number in `test_meshless_galerkin_mfg.py`'s docstring (6.27e-02 →
6.0468e-02; the assertion is unaffected).
