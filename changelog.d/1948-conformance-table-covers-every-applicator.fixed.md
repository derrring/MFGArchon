The `(applicator, BCType)` conformance table now covers all six applicators, not three, and
`GraphApplicator` is indexed by the alphabet it actually speaks. The table goes from 24 cells to 50
— 40 in the `BCType` product over the five applicators that speak it, plus a separate
(`GraphBCType` × `field_type`) product of 10 — and records six silent cells, each marked
`xfail(strict=True)` so that fixing one turns the table red rather than quiet.

`GraphApplicator` gets its own product because its alphabet is `GraphBCType`, disjoint from
`BCType`, and it gets a second axis because **three of its five arms** branch on `field_type` inside
themselves: a Dirichlet pin belongs to the value field, an absorbing node does opposite things to
the two halves, and a source injects only into the density. NEUMANN and CUSTOM do not read it, so
four of the ten cells are two labels over one code path and the ten cells cover seven distinct
paths — recorded in the file rather than left to read as coverage it does not have.

Three of the five rows of the single-`field_type` table did not say what their label claimed, and
the three do not share one cause. SOURCE was **wrong** (it applies on the density) and DIRICHLET was
**incomplete** (its verdict was true; its silent density half was unmarked and out of view) — both
repaired by the axis. CUSTOM was wrong for a different reason: its arm never reads `field_type`, so
only giving it the callable its API documents moves it, and the axis alone would leave six cells
silent rather than four. Measured: break the source injection into the density and exactly one test
in the whole suite fails, one of the new cells; the single-axis table stays green because it had
already written that cell off as silent.

The table's own exemption mechanism did not work. `_KNOWN_SILENT` cells were skipped with the
imperative `pytest.xfail()`, which aborts before the assertion and so can never report XPASS —
measured against this repository's `xfail_strict = true`, a marker-based xfail on a passing test
reports `FAILED [XPASS(strict)]` while the imperative form reports a silent `xfailed`. The comment
promising that "fixing one turns this table red" was therefore false. Exemptions now go through the
parametrisation as `xfail(strict=True)`, and removing a cell from the set makes it fail.
