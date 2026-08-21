Every ratchet now carries a positive control, and the gate runs them. A ratchet whose measurement
has gone blind reports a stable or *falling* count and reads exactly like success —
`check_internal_deprecation`'s own docstring records that its predecessor printed
`Total deprecated symbols: 0` while 72 were live, with CI green over it. Added two-sided controls to
`check_fail_fast` (every category fires on a violation file and stays silent on a clean one whose
docstrings mention the same words) and `check_assertion_strength` (flags the weak control, leaves an
exact-value comparison alone), and a named-sentinel control to `check_internal_deprecation` (one live
deprecation per counted kind, since its population is the real package and there is no clean tree to
assert silence over). `check_doc_api` and `capability_matrix` already had controls **that nothing
ever ran** — the gate now invokes all five: four fast ones (7s) before the ratchets they guard, and
the capability self-test (92s) in the slow tier beside its matrix.
