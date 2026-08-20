`source_term` now has one calling convention across the FP solvers (#2019).

`FPFVMSolver` passed `geometry.meshgrid()` — a `(d, *shape)` tuple — and accepted only a grid-shaped
return. `FPFDMSolver` and the whole HJB side pass `geometry.get_spatial_grid()`, an `(N, d)` point
array, and ravel/reshape what comes back. So one callback could not serve both: a source written
against the convention `BaseHJBSolver.solve_hjb_system` documents ("x has shape (N, d)") returned an
`(N,)` array to FVM and hit

```
ValueError: operands could not be broadcast together with shapes (21,21) (882,)
```

where 882 = 2 × 441 is the two coordinate planes flattened together. FVM is moved onto the documented
convention on both sides — argument and return — with a fail-loud size check rather than a broadcast.

**Why it survived, and why the new tests are 2-D.** In 1-D the two conventions coincide:
`meshgrid()` gives one `(21,)` array and `get_spatial_grid()` gives `(21, 1)`, so a raveling callback
accepts both. The fork is expressible only in `d ≥ 2` — the rule `AGENTS.md` states as "the dimension
must be able to express the property under test", and a test records that no 1-D test could have
caught it.

**A second divergence found and deliberately left alone.** FVM evaluates the source at `k*dt` where
FDM uses `(k+1)*dt` and calls it implicit. Measured by MMS on a pure-diffusion manufactured solution,
the two are indistinguishable at achievable resolutions: the first refinement gives order 0.910 at
`k*dt` against 0.847 at `t_{k+1}`, and both then degrade — as does an FDM control, to 0.343 — because
this scheme is first order in **space** (measured 1.026 / 1.004 / 0.978) and the spatial floor
dominates before the temporal order is visible. Changing it would be a behaviour change with no
evidence behind it, so the measurement is recorded at the line instead.

Mutation-checked: reverting to `meshgrid()` fails both FVM cases; dropping the `reshape` fails the
one that checks the source actually moves the answer.
