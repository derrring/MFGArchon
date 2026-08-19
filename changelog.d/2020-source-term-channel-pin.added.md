A behavioural pin that `source_term` is either honoured or refused, never silently discarded.

`inspect.signature` answers "does this callable name the parameter", which is coarser than "does a
source reach the discretisation", and the repo contains both ways the two diverge: `HJBWENOSolver`
names the parameter and still refuses it in multi-D, while four solvers declare `**kwargs` and
swallow it whole. A census keyed on either signature property misclassifies one group or the other,
which is how #1991's table came to carry a row that was wrong when it was written.

The new test passes a real source and watches the answer. Measured at the time of writing:

| solver | source calls | max&#124;with − without&#124; |
|---|---|---|
| `HJBFDMSolver` | 5 | 1.000000e+00 |
| `FPFDMSolver` | 5 | 1.000000e+00 |
| `MeshlessGalerkinHJBSolver` | 0 | 0.000000e+00 |
| `MeshlessGalerkinFPSolver` | 0 | 0.000000e+00 |
| `HJBFEMSolver` | 0 | 0.000000e+00 |
| `FPFEMSolver` | 0 | 0.000000e+00 |

The last four rows are a **recorded defect pin, not a specification** (#2020): they assert the
current silent-discard so that fixing it — by honouring the source or by refusing it — trips the
assertion, and the failure message says to move the solver out of the list.

A second test classifies every solver in `mfgarchon.alg` by MRO-resolved signature and pins the set
that swallows through `**kwargs`, so a new solver inheriting a `**kwargs` solve method cannot join
the defect silently. Classification is by MRO and not by each class's own `__dict__`: three of the
six define nothing themselves and resolve to a concrete intermediate that already dropped the
parameter.

Scope, established by survey rather than assumed: the coupling layer is not the hazard —
`resolve_volatility_kwarg` treats `**kwargs` as not accepting a parameter and raises, and the
`source_term` branch beside it has done the same since #1424. The hazard is the direct
`solve_*_system` call, and no test in the suite currently drives a manufactured solution through a
swallowing solver.

Four mutants confirm the pin discriminates: reclassifying a swallower as honouring, reclassifying a
threader as swallowing, dropping a name from the pinned set, and making the injected source
identically zero each fail the expected row.
