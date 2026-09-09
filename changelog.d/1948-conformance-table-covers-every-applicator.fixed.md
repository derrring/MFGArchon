The `(applicator, BCType)` conformance table now covers all six applicators, not three, and
`GraphApplicator` is indexed by the alphabet it actually speaks. The table goes from 24 cells to 50
— 40 in the `BCType` product over the five applicators that speak it, plus a separate
(`GraphBCType` × `field_type`) product of 10 — and records six silent cells, each marked
`xfail(strict=True)` so that fixing one turns the table red rather than quiet.

`GraphApplicator` gets its own product because its alphabet is `GraphBCType`, disjoint from
`BCType`, and it gets two axes because its `apply` branches on `field_type` inside every arm: a
Dirichlet pin belongs to the value field, an absorbing node does opposite things to the two halves,
and a source injects only into the density. A single-`field_type` table measured the fixture's
default rather than the class, and carried three of five rows wrong — it recorded SOURCE and CUSTOM
as silent when each applies given the right field type and the value its own API documents, and
recorded DIRICHLET as applying while its silent density half stayed out of view. Measured: break
the source injection into the density and exactly one test in the whole suite fails, one of the new
cells; the single-axis table stays green because it had already written that cell off as silent.

The table's own exemption mechanism did not work. `_KNOWN_SILENT` cells were skipped with the
imperative `pytest.xfail()`, which aborts before the assertion and so can never report XPASS —
measured against this repository's `xfail_strict = true`, a marker-based xfail on a passing test
reports `FAILED [XPASS(strict)]` while the imperative form reports a silent `xfailed`. The comment
promising that "fixing one turns this table red" was therefore false. Exemptions now go through the
parametrisation as `xfail(strict=True)`, and removing a cell from the set makes it fail.
