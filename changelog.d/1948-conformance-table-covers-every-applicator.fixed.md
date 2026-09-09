The `(applicator, BCType)` conformance table now covers all six applicators, not three. Extending
it to `InterpolationApplicator`, `ParticleApplicator` and `GraphApplicator` takes the table from 24
cells to 48 and surfaces ten silent cells: `GraphApplicator` returns the field untouched for every
one of the eight `BCType` members, because its own alphabet is `GraphBCType` and its `apply` has no
terminal `else` (the same gap #1940 closed for that method's `field_type` parameter), and
`InterpolationApplicator` is silent on both `EXTRAPOLATION_*`.

The table's own exemption mechanism did not work. `_KNOWN_SILENT` cells were skipped with the
imperative `pytest.xfail()`, which aborts before the assertion and so can never report XPASS —
measured against this repository's `xfail_strict = true`, a marker-based xfail on a passing test
reports `FAILED [XPASS(strict)]` while the imperative form reports a silent `xfailed`. The comment
promising that "fixing one turns this table red" was therefore false. Exemptions are now applied
through the parametrisation as `xfail(strict=True)`, and removing a cell from the set makes it fail.
