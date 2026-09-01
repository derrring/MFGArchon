`FictitiousPlayIterator` and `BlockIterator` silently ignored a problem's `source_term_hjb`,
`source_term_fp` and `nonlocal_operator`: neither loop composed one, so the fail-loud guard added
in #1424 was never reached and the solve returned a value function bit-identical to one for a
problem carrying no source at all. Measured on a fixture with a constant source of 50.0, both
reported `max|dU| = 0.000000e+00` with the source callable invoked 0 times, against
`FixedPointIterator`'s 8.75 over 15 invocations. `BaseCouplingIterator._build_hjb_kwargs` /
`_build_fp_kwargs` now compose the source themselves from the iterates, which are required
parameters — the omission is no longer expressible — and `FixedPointIterator`'s two composition
delegates are deleted rather than left beside the new owner. (#2207)

Consequence, and it is a behaviour change beyond the two loops: pairing either of those loops with
an FP or HJB solver that does not accept `source_term` — six of the nine FP solvers, and
`HJBHowardSolver` — and a problem that defines a source now raises `NotImplementedError` where it
previously ran and silently solved a different equation. That is #1424's contract reaching the
loops it could not reach before. `BlockIterator`'s strict-adjoint FP path (`adjoint_mode != "off"`)
assembles and steps its own operator instead of calling the FP solver, so it has no route for a
source at all and now refuses one explicitly rather than dropping it. (#2207)
