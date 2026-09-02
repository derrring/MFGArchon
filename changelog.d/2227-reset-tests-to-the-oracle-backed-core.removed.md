**243 test files and 58,785 lines deleted.** `tests/` goes from 463 files / 135,016 lines to 220 /
76,231. Git history is the archive, the same disposition #1710 gave `archive/` and the one that
removed `alg/neural/` and `alg/reinforcement/`.

**Why, in one line:** a suite you cannot audit is not an asset, and this one could not be audited —
5,632 test functions, and reading them file by file was costed at roughly six days of full-time
review.

**What replaced "audit the tests" as the question.** The `[Principle] Conventions Index` names **35**
load-bearing conventions. `scripts/discrimination_baseline.json` carries a falsifiable mutation for
**24** of them, each with an `owner` field naming the convention it breaks. So the work unit is 35,
not 5,632 — three orders of magnitude — and the convention↔test map already existed in the
repository rather than needing to be built.

Measured before the deletion: **every one of the 24 is defended.** `kill_count` runs 2 … 168, median
22, **no zeros and no single points of failure**. The three thinnest, at 2 each, are the ones worth
adding to rather than anything worth removing:

- absent BC defaults to clamp/absorbing (#1698)
- `SolverResult.mass_conservation_error` is drift from the initial mass, not deviation from 1
- Picard convergence requires **both** the relative and the absolute criterion

## The keep-set was built from measurement, then controlled

Not from a keyword grep. The union of: every file containing a test that **kills a mutation**
(`discrimination_killmatrix.json`, 125 files — measured, not matched); files carrying an **external
oracle**; **defect pins** with retirement conditions; tests that **guard the instruments** in
`scripts/`; the ratchet's own self-test; `conftest.py` and `__init__.py`.

The instrument-guard category exists because the **positive control caught its absence**: a first
keep-set would have deleted `tests/unit/test_capability_warnings.py`, which carries the `== 12`
defect pin for #1878 and the `library_said` machinery. That is the contamination this repository has
recorded in five of five mechanical criteria, caught here by running the control rather than by
trusting the criterion.

**The deletion's own acceptance test, run after:** 613 of the 614 killer node IDs survive, **0**
killer files deleted, and **no convention's `kill_count` reached zero**. One node's function name no
longer resolves, taking `particle_mass_counting_measure` from 13 to 12. Control: a file known to be
deleted reports missing.

**Remaining suite:** 3,583 passed, 3 skipped, 20 xfailed in 150 s on the gate's marker set.

## Two mechanical consequences, both recorded rather than worked around

`scripts/citation_baseline.json` re-recorded — 42 citations pointed at deleted files. That moved the
standing backlog from 13 rows to 11, and the number is hand-written in **three** places
(`check_citations.py` twice, one changelog fragment); its own test says so while failing: *"Every
copy must be edited together, which is why there should not be three."* All three updated. The
statement remains true: the 11 are the survivors of the 13 that were read by hand.

The discrimination sweep was **not** re-run end to end. It plants 24 mutations and runs the suite for
each, roughly 40 minutes, and a run killed by a harness timeout leaves a mutant in the tree — its own
guard says the `finally` "survives neither SIGKILL nor a harness timeout" (#1849). That happened
twice today. The killer-survival check above answers the same question in seconds and without
mutating anything. (#2227)
