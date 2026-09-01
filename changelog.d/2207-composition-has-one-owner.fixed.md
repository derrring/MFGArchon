`FictitiousPlayIterator` and `BlockIterator` silently ignored a problem's `source_term_hjb`,
`source_term_fp` and `nonlocal_operator`: neither loop composed one, so the fail-loud guard added
in #1424 was never reached and the solve returned a value function bit-identical to one for a
problem carrying no source at all. Measured on a fixture with a constant source of 50.0, both
reported `max|dU| = 0.000000e+00` with the source callable invoked 0 times, against
`FixedPointIterator`'s 8.75 over 15 invocations. `BaseCouplingIterator._build_hjb_kwargs` /
`_build_fp_kwargs` now compose the source themselves from the iterates, which are required
parameters — the omission is no longer expressible — and `FixedPointIterator`'s two composition
delegates are deleted rather than left beside the new owner. (#2207)
